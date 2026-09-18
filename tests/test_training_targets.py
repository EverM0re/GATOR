import json
import os
import tempfile
from types import SimpleNamespace

from file_router.training.dataset import RouterDataset
from file_router.training.trainer import RouterTrainer


def _group(group_id, cost):
    return {
        "group_id": group_id,
        "root_node_id": group_id,
        "node_ids": [group_id],
        "node_class": "text_span",
        "level": 0,
        "redundancy_cluster_id": "doc:p0",
        "base_score": 0.5,
        "token_equivalent_cost": cost,
        "source_doc_id": "doc",
    }


def _batch(minimal):
    record = {
        "query_id": "q",
        "query": "When did it happen?",
        "candidate_groups": [_group("a", 10), _group("b", 20)],
        "gold_supporting_groups": ["a", "b"],
        "minimal_sufficient_groups": minimal,
        "modality_bias": {"active_modality": "text", "confidence": 0.3,
                          "apply_bias": False},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        dataset = RouterDataset(handle.name, use_minimal=True, max_cost=6000)
        return dataset.to_tensors(dataset.load())[0]


def test_multiple_minimal_targets_remain_multi_hot():
    batch = _batch(["a", "b"])
    assert batch["minimal_labels"].tolist() == [1.0, 1.0]


def test_empty_minimal_target_does_not_fall_back_to_gold():
    batch = _batch([])
    assert batch["minimal_labels"].tolist() == [0.0, 0.0]


def test_trainer_uses_padded_minibatches_and_writes_metadata():
    records = []
    for index in range(12):
        records.append({
            "query_id": f"q{index}", "query": "How many?",
            "candidate_groups": [_group("a", 10), _group("b", 20)],
            "gold_supporting_groups": ["a"],
            "minimal_sufficient_groups": ["a"],
            "modality_bias": {},
        })
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "train.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        cfg = SimpleNamespace(
            cost=SimpleNamespace(max_cost_normalizer=6000.0),
            router=SimpleNamespace(learned_retrieval_prior=1.0),
            trainer=SimpleNamespace(
                seed=42, use_minimal_sufficient=True, input_dim=28,
                hidden_dim=8, device="cpu", learning_rate=1e-3,
                margin=0.5, lambda_margin=0.2, validation_ratio=0.2,
                early_stopping_patience=2, early_stopping_min_delta=1e-4,
                batch_size=4, num_epochs=2, lambda_route=1.0,
                lambda_cost=0.5, lambda_level=0.4,
            ),
        )
        RouterTrainer(cfg).train(path, directory)
        with open(os.path.join(directory, "router_meta.json"),
                  encoding="utf-8") as handle:
            meta = json.load(handle)
    assert meta["batch_size"] == 4
    assert meta["learned_retrieval_prior"] == 1.0


def test_trainer_records_per_epoch_history_and_stopping_reason():
    """The paper reports convergence, so the run must record how it ended.

    `early_stopped` distinguishes a run that patience terminated (evidence of
    convergence) from one the epoch budget cut off (no such evidence), which is
    the claim a reviewer checks.
    """
    records = []
    for index in range(12):
        records.append({
            "query_id": f"q{index}", "query": "How many?",
            "candidate_groups": [_group("a", 10), _group("b", 20)],
            "gold_supporting_groups": ["a"],
            "minimal_sufficient_groups": ["a"],
            "modality_bias": {},
        })
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "train.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        cfg = SimpleNamespace(
            cost=SimpleNamespace(max_cost_normalizer=6000.0),
            router=SimpleNamespace(learned_retrieval_prior=1.0),
            trainer=SimpleNamespace(
                seed=42, use_minimal_sufficient=True, input_dim=28,
                hidden_dim=8, device="cpu", learning_rate=1e-3,
                margin=0.5, lambda_margin=0.2, validation_ratio=0.2,
                early_stopping_patience=2, early_stopping_min_delta=1e-4,
                batch_size=4, num_epochs=3, lambda_route=1.0,
                lambda_cost=0.5, lambda_level=0.4,
            ),
        )
        RouterTrainer(cfg).train(path, directory)
        with open(os.path.join(directory, "router_meta.json"),
                  encoding="utf-8") as handle:
            meta = json.load(handle)

    history = meta["history"]
    assert len(history) == meta["epochs_run"] >= 1
    assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
    # Diagnostics the training figure draws on.
    for field in ("train_loss", "grad_norm", "route_loss", "learning_rate"):
        assert all(h[field] is not None for h in history), field
    # Three epochs cannot exhaust patience 2 without two stale epochs, so this
    # run ends at its budget and must not claim early stopping.
    assert meta["early_stopped"] is (meta["epochs_run"] < 3)
