"""Create and run a reproducible small-scale File_Router validation.

This orchestrator writes an isolated run directory, derives an effective YAML
config, executes preparation/training/evaluation stages, and tees every stage
to run.log. Secrets are read from environment variables and never written to
the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_number(name: str, default, cast):
    """Read an optional numeric override.

    An unset or empty variable keeps the YAML value, so run_experiment.sh can
    export every knob unconditionally and still mean "use the config".
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return cast(value.strip())
    except ValueError:
        print(f"[run_validation] ignoring non-numeric {name}={value!r}")
        return default


def _auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _absolute_from_project(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return str(candidate.resolve())


def resolved_api_key(config_path: str) -> str:
    """The key the stages must use, resolved the same way the VLM client does.

    Config normally wins, because each endpoint carries its own credential and
    a stale exported key must not silently replace all of them.

    The exception is an explicitly overridden endpoint: the backbone ablation
    sets LLM_BASE_URL and LLM_API_KEY together to point at a different provider,
    and pairing that host with the config's key authenticates the wrong service.
    A key supplied alongside a base_url override therefore wins, since it was
    chosen for that host.
    """
    try:
        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        llm = raw.get("llm") or {}
        from_config = str(llm.get("api_key") or "")
        config_url = str(llm.get("base_url") or "")
    except Exception:  # noqa: BLE001
        from_config, config_url = "", ""

    env_key = os.environ.get("LLM_API_KEY", "")
    env_url = os.environ.get("LLM_BASE_URL", "")
    if env_key and env_url and env_url != config_url:
        return env_key
    return (from_config
            or env_key
            or os.environ.get("OPENAI_API_KEY", ""))


def _effective_config(args, run_dir: Path) -> dict:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw["datasets"]["subset_qa"] = args.subset
    raw["datasets"]["subset_docs"] = args.docs
    raw["evaluation"]["max_questions"] = args.eval_per_dataset
    raw["trainer"]["num_epochs"] = args.epochs
    raw["trainer"]["device"] = _auto_device(args.device)
    raw["trainer"].setdefault("opd", {})
    raw["trainer"]["opd"]["enabled"] = _env_bool(
        "OPD_ENABLED", bool(raw["trainer"]["opd"].get("enabled", True)))
    raw["trainer"]["opd"]["run_ab"] = _env_bool(
        "OPD_RUN_AB", bool(raw["trainer"]["opd"].get("run_ab", True)))
    raw["split"]["seed"] = int(os.environ.get(
        "SPLIT_SEED", raw["split"].get("seed", 42)))
    # Varying SPLIT_SEED alone does not give independent runs: with a prebuilt
    # store the split is reused, so every seed trains the same model from the
    # same initialisation and the "3 seeds" collapse to one number. TRAIN_SEED
    # varies the initialisation and batch order, which is what seed variance in
    # the paper is meant to measure.
    raw["trainer"]["seed"] = int(os.environ.get(
        "TRAIN_SEED", raw["trainer"].get("seed", 42)))
    raw["trainer"]["architecture"] = os.environ.get(
        "ROUTER_ARCH", raw["trainer"].get("architecture", "mlp2"))

    # Retrieval / selector overrides.  campaign_20260823_010042 showed the
    # binding constraint was page-level candidate recall (page_recall@5 = 0.40)
    # and a selector that kept only 1.75 groups, so these are the knobs a sweep
    # needs to reach without editing the YAML.
    retrieval = raw.setdefault("retrieval", {})
    retrieval["topk_text"] = _env_number(
        "TOPK_TEXT", retrieval.get("topk_text", 30), int)
    retrieval["topk_visual"] = _env_number(
        "TOPK_VISUAL", retrieval.get("topk_visual", 15), int)
    retrieval["max_candidates"] = _env_number(
        "MAX_CANDIDATES", retrieval.get("max_candidates", 100), int)

    router = raw.setdefault("router", {})
    router["min_distinct_pages"] = _env_number(
        "MIN_DISTINCT_PAGES", router.get("min_distinct_pages", 0), int)
    router["min_relative_probability"] = _env_number(
        "MIN_RELATIVE_PROBABILITY",
        router.get("min_relative_probability", 0.05), float)
    router["submod_gamma"] = _env_number(
        "SUBMOD_GAMMA", router.get("submod_gamma", 0.8), float)
    # The screenshot price is the one number in the cost model that is assumed
    # rather than measured, and every reported saving is denominated in it.
    # PDF_PAGE_TOKEN_COST re-prices it so the conclusions can be checked against
    # a different assumption.
    # An arm that re-prices the ladder must not share a store with one that did
    # not: costs are baked in at ingest, so a shared store silently serves the
    # price it was built with.
    if os.environ.get("STORE_DIR"):
        raw["paths"]["store_dir"] = os.environ["STORE_DIR"]

    if os.environ.get("PDF_PAGE_TOKEN_COST"):
        raw["cost"]["pdf_page_token_budget"] = int(
            os.environ["PDF_PAGE_TOKEN_COST"])

    # DISABLE_TIERS="caption" or "caption,fulltext"; empty keeps the YAML value.
    disable_tiers = os.environ.get("DISABLE_TIERS")
    if disable_tiers is not None and disable_tiers.strip():
        router["disable_tiers"] = [
            part.strip().lower()
            for part in disable_tiers.split(",") if part.strip()]
    router["expensive_tier_cost"] = _env_number(
        "EXPENSIVE_TIER_COST", router.get("expensive_tier_cost", 0.0), float)
    router["expensive_tier_max_rank"] = _env_number(
        "EXPENSIVE_TIER_MAX_RANK", router.get("expensive_tier_max_rank", -1), int)
    router["expensive_tier_min_probability_share"] = _env_number(
        "EXPENSIVE_TIER_MIN_PROB_SHARE",
        router.get("expensive_tier_min_probability_share", 0.0), float)

    raw["encoders"]["text"]["backend"] = os.environ.get(
        "TEXT_BACKEND", raw["encoders"]["text"]["backend"])
    raw["encoders"]["visual"]["backend"] = os.environ.get(
        "VISUAL_BACKEND", raw["encoders"]["visual"]["backend"])
    encoder_device = os.environ.get("ENCODER_DEVICE", _auto_device(args.device))
    raw["encoders"]["text"]["device"] = encoder_device
    raw["encoders"]["visual"]["device"] = encoder_device

    raw["llm"]["backend"] = os.environ.get(
        "LLM_BACKEND", raw["llm"].get("backend", "stub"))
    raw["llm"]["base_url"] = os.environ.get(
        "LLM_BASE_URL", raw["llm"].get("base_url", ""))
    raw["llm"]["model"] = os.environ.get(
        "LLM_MODEL", raw["llm"].get("model", ""))
    raw["llm"]["vision"] = _env_bool(
        "LLM_VISION", bool(raw["llm"].get("vision", False)))
    # The effective config is written into the run directory and archived, so
    # the key is stripped from it rather than persisted.  It is handed to the
    # stages through the environment instead (see _stage_env), which keeps the
    # secret out of the artefact without leaving the stages unauthenticated.
    raw["llm"]["api_key"] = ""
    raw["evaluation"].setdefault("lm_judge", {})
    raw["evaluation"]["lm_judge"]["enabled"] = _env_bool(
        "LM_JUDGE", bool(raw["evaluation"]["lm_judge"].get("enabled", True)))
    # Caption backend is an ingestion-time choice, so switching it requires a
    # re-ingest; exposing it here lets the caption-quality ablation run without
    # editing the YAML between arms.
    raw["ingestion"]["caption"]["backend"] = os.environ.get(
        "CAPTION_BACKEND", raw["ingestion"]["caption"].get("backend", "heuristic"))
    raw["ingestion"]["caption"]["vlm_for_image_only"] = _env_bool(
        "VLM_CAPTION_IMAGE_ONLY",
        bool(raw["ingestion"]["caption"].get("vlm_for_image_only", True)),
    )

    raw["paths"]["log_dir"] = str(run_dir / "internal_logs")
    raw["paths"]["training_data_path"] = str(run_dir / "router_training.jsonl")
    if args.reuse_model:
        raw["paths"]["router_model_dir"] = _absolute_from_project(args.reuse_model)
        raw["router"]["router_model_path"] = raw["paths"]["router_model_dir"]
    else:
        raw["paths"]["router_model_dir"] = str(run_dir / "router_model")
        raw["router"]["router_model_path"] = ""
    return raw


class StageRunner:
    def __init__(self, log_path: Path, extra_env: dict | None = None):
        self.log_path = log_path
        self.statuses = []
        # Secrets stripped from the persisted config are passed to the stages
        # here instead, so the artefact stays clean and the stages stay
        # authenticated.
        self.extra_env = dict(extra_env or {})

    def run(self, name: str, command: list[str]) -> None:
        started = datetime.now()
        header = (f"\n{'=' * 72}\nSTAGE: {name}\nCOMMAND: "
                  f"{shlex.join(command)}\n{'=' * 72}\n")
        print(header, end="", flush=True)
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(header)
            stage_env = os.environ.copy()
            stage_env.update(self.extra_env)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=stage_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        elapsed = (datetime.now() - started).total_seconds()
        self.statuses.append({"stage": name, "exit_code": code,
                              "seconds": round(elapsed, 3)})
        if code in (130, 143):
            raise KeyboardInterrupt(f"stage {name!r} interrupted")
        if code != 0:
            raise RuntimeError(f"stage {name!r} failed with exit code {code}")


def _write_failure_report(run_dir: Path, message: str, statuses: list[dict]) -> None:
    lines = [
        "# GATOR validation run failed",
        "",
        f"> {message}",
        "",
        "## Stages executed",
        "",
        "| Stage | Exit code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in statuses:
        lines.append(f"| {row['stage']} | {row['exit_code']} | {row['seconds']} |")
    lines += [
        "",
        "Please also attach `run.log` from this directory; it contains the full error.",
        "",
    ]
    with open(run_dir / "RUN_FAILED.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _append_stage_timings(report_path: Path, statuses: list[dict]) -> None:
    token_summary = {}
    summary_path = report_path.parent / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                token_summary = json.load(f).get("token_and_time", {})
        except Exception:  # noqa: BLE001
            token_summary = {}

    def token_totals(section: str) -> tuple[int, int, int]:
        rows = token_summary.get(section, {})
        prompt = sum(int(row.get("prompt_tokens", 0)) for row in rows.values())
        completion = sum(int(row.get("completion_tokens", 0)) for row in rows.values())
        return prompt, completion, prompt + completion

    ingest_tokens = token_totals("ingestion_vlm")
    evaluation_tokens = token_totals("evaluation_vlm")
    lines = [
        "",
        "## 14. Per-stage pipeline timing",
        "",
        "| Stage | Exit code | Seconds | Input Tokens | Output Tokens | Total Tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in statuses:
        if "ingest" in row["stage"]:
            tokens = ingest_tokens
        elif "evaluate" in row["stage"]:
            tokens = evaluation_tokens
        else:
            tokens = (0, 0, 0)
        lines.append(
            f"| {row['stage']} | {row['exit_code']} | {row['seconds']} | "
            f"{tokens[0]} | {tokens[1]} | {tokens[2]} |")
    lines.append("")
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _record_hardware(run_dir: Path) -> None:
    """Write the accelerator this run actually used.

    The paper reports the hardware, and a claim about it should come from the
    machine rather than from memory: nothing in the logs previously recorded
    which GPU served a run.
    """
    info_lines = []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            info_lines = [ln.strip() for ln in out.stdout.strip().splitlines()]
    except Exception:  # noqa: BLE001
        pass
    if not info_lines:
        try:
            import torch
            if torch.cuda.is_available():
                info_lines = [
                    f"{torch.cuda.get_device_name(i)}, "
                    f"{torch.cuda.get_device_properties(i).total_memory // 2**20} MiB"
                    for i in range(torch.cuda.device_count())]
            else:
                info_lines = ["no CUDA device visible (CPU run)"]
        except Exception:  # noqa: BLE001
            info_lines = ["hardware unknown: nvidia-smi and torch both unavailable"]
    text = "\n".join(info_lines) + "\n"
    (run_dir / "hardware.txt").write_text(text, encoding="utf-8")
    for line in info_lines:
        print(f"[hardware] {line}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/file_router.yaml")
    parser.add_argument("--subset", type=int, default=30,
                        help="raw QA cap per dataset")
    parser.add_argument("--docs", type=int, default=8,
                        help="document cap where supported")
    parser.add_argument("--eval-per-dataset", type=int, default=8,
                        help="test questions per dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--run-root", default="validation_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--reuse-model", default="",
                        help="skip training and load an existing router model dir")
    args = parser.parse_args()

    if args.subset <= 0:
        parser.error("--subset must be positive for a validation run")
    if args.eval_per_dataset <= 0:
        parser.error("--eval-per-dataset must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"validation_{timestamp}"
    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_DIR / run_root
    run_dir = (run_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "internal_logs").mkdir()
    _record_hardware(run_dir)

    config = _effective_config(args, run_dir)
    config_path = run_dir / "effective_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(PROJECT_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "arguments": vars(args),
        "secrets": {
            "LLM_API_KEY_present": bool(os.environ.get("LLM_API_KEY") or
                                        os.environ.get("OPENAI_API_KEY")),
            "api_key_persisted": False,
        },
        "llm": {
            "backend": config["llm"]["backend"],
            "base_url": config["llm"]["base_url"],
            "model": config["llm"]["model"],
            "vision": config["llm"]["vision"],
        },
        "encoders": {
            "text": config["encoders"]["text"]["backend"],
            "visual": config["encoders"]["visual"]["backend"],
            "device": config["encoders"]["visual"].get("device", "auto"),
        },
        "evaluation": {
            "lm_judge": config["evaluation"]["lm_judge"]["enabled"],
            "split_seed": config["split"]["seed"],
        },
        "training": {
            "opd_enabled": config["trainer"]["opd"]["enabled"],
            "opd_run_ab": config["trainer"]["opd"]["run_ab"],
            "opd_divergence": config["trainer"]["opd"].get("divergence"),
            "lambda_opd": config["trainer"].get("lambda_opd", 0.0),
        },
        "stages": [],
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    api_key = resolved_api_key(args.config)
    if not api_key:
        print("[run_validation] WARNING: no LLM api_key in config or "
              "environment; stages will authenticate as EMPTY")
    runner = StageRunner(run_dir / "run.log",
                         extra_env={"LLM_API_KEY": api_key} if api_key else None)
    py = sys.executable
    try:
        if not args.skip_download:
            runner.run("download small subsets", [
                py, "-m", "scripts.download_data", "--config", str(config_path),
                "--subset", str(args.subset), "--docs", str(args.docs),
            ])
        if not args.skip_build:
            runner.run("build unified data and deterministic splits", [
                py, "-m", "scripts.build_unified", "--config", str(config_path),
                # Must match the download stage: without these the build caps
                # every dataset at the config's subset_docs and silently
                # discards most of what was just fetched.
                "--docs", str(args.docs), "--subset", str(args.subset),
            ])
        runner.run("validate unified data scale", [
            py, "-m", "scripts.check_data_scale", "--config", str(config_path),
            "--subset", str(args.subset), "--docs", str(args.docs),
            "--eval-per-dataset", str(args.eval_per_dataset),
        ])
        if not args.skip_ingest:
            runner.run("ingest three-tier evidence stores", [
                py, "-m", "scripts.ingest", "--config", str(config_path),
            ])
        if not args.reuse_model:
            runner.run("train learned router", [
                py, "-m", "scripts.train", "--config", str(config_path),
            ])
        runner.run("evaluate and write comprehensive report", [
            py, "-m", "scripts.evaluate_validation", "--config", str(config_path),
            "--run-dir", str(run_dir),
        ])
    except KeyboardInterrupt as exc:
        manifest["stages"] = runner.statuses
        manifest["status"] = "interrupted"
        manifest["error"] = repr(exc)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        _write_failure_report(run_dir, "experiment interrupted", runner.statuses)
        print(f"\nValidation interrupted. Partial run: {run_dir}", flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001
        manifest["stages"] = runner.statuses
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        _write_failure_report(run_dir, str(exc), runner.statuses)
        print(f"\nValidation failed. See: {run_dir / 'RUN_FAILED.md'}")
        return 2

    manifest["stages"] = runner.statuses
    manifest["status"] = "complete"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            manifest["token_and_time"] = json.load(f).get("token_and_time", {})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    _append_stage_timings(run_dir / "REPORT.md", runner.statuses)
    print("\nValidation complete")
    print(f"Report : {run_dir / 'REPORT.md'}")
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
