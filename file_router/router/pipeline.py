"""End-to-end router: recall → score → select → materialize context → generate."""

from __future__ import annotations

from typing import List

from ..schemas import EvidenceGroup, ModalityBias, RouterResult
from .scorer import RuleBasedScorer, LearnedScorer
from .selector import build_selector


class RouterPipeline:
    def __init__(self, cfg, retriever, nodes, vlm, exporter=None):
        self.cfg = cfg
        self.retriever = retriever
        self.nodes = nodes
        self.vlm = vlm
        self.exporter = exporter

        if cfg.router.use_learned_scorer:
            path = (cfg.router.router_model_path
                    if cfg.router.router_model_path else cfg.paths.router_model_dir)
            self.scorer = LearnedScorer(cfg.router, cfg.trainer, path)
        else:
            self.scorer = RuleBasedScorer(cfg.router)
        self.selector = build_selector(cfg.router)

    # ------------------------------------------------------------ answer
    def answer(self, question: str) -> RouterResult:
        groups, bias = self.retriever.recall(question)

        if not groups:
            ans = self.vlm.generate_answer(question, context="(no retrieved evidence)")
            return RouterResult(question=question, answer=ans,
                                selected_groups=[], candidate_count=0,
                                strategy=self.cfg.router.strategy,
                                total_cost=0.0,
                                modality_bias={"active_modality": bias.active_modality,
                                               "confidence": bias.confidence,
                                               "apply_bias": bias.apply_bias},
                                fell_back=True)

        # Score
        self.scorer.score(
            groups, self.cfg.cost.max_cost_normalizer,
            bias=bias, question=question,
        )

        # Select
        selected = self.selector.select(groups)
        if not selected:
            selected = sorted(groups, key=lambda g: -g.router_probability)[:1]

        # Materialize context and generate
        context, image_paths, total_cost = self._materialize(selected)
        answer = self.vlm.generate_answer(
            question, context=context, image_paths=image_paths)

        # Export training sample
        if self.cfg.router.export_training_data and self.exporter is not None:
            self.exporter.export(
                question=question, answer=answer,
                candidate_groups=groups, selected_groups=selected,
                bias=bias,
            )

        return RouterResult(
            question=question, answer=answer,
            selected_groups=[g.group_id for g in selected],
            candidate_count=len(groups),
            strategy=self.cfg.router.strategy,
            total_cost=total_cost,
            modality_bias={"active_modality": bias.active_modality,
                           "confidence": bias.confidence,
                           "apply_bias": bias.apply_bias},
        )

    # ------------------------------------------------------------ helpers
    def _materialize(self, groups: List[EvidenceGroup]):
        """Turn selected groups into text plus actual image attachments."""
        total_cost = 0.0
        lines: List[str] = []
        image_paths: List[str] = []
        for g in groups:
            total_cost += g.token_equivalent_cost
            for nid in g.node_ids:
                n = self.nodes.get(nid)
                if n is None:
                    continue
                if n.text:
                    tag = f"[{n.node_class} L{n.level} doc={n.source_ref.doc_id}"
                    if n.source_ref.page_num is not None:
                        tag += f" p{n.source_ref.page_num}"
                    tag += "]"
                    lines.append(f"{tag} {n.text}")
                elif n.image_path:
                    lines.append(f"[image doc={n.source_ref.doc_id} path={n.image_path}]"
                                 " (see file for visual content)")
                    image_paths.append(n.image_path)
                elif n.page_image_path:
                    lines.append(f"[pdf_page doc={n.source_ref.doc_id} "
                                 f"p{n.source_ref.page_num} path={n.page_image_path}]"
                                 " (see file for visual content)")
                    image_paths.append(n.page_image_path)
        return "\n\n".join(lines), list(dict.fromkeys(image_paths)), total_cost
