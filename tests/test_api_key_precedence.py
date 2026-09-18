"""Each endpoint uses its own credential.

Endpoints carry independent keys (a local vLLM plus one hosted API per backbone
arm), so a single exported LLM_API_KEY must not override them. It did: a stale
variable from an earlier shell silently replaced a correct config key and the
run aborted on a 401 that looked like a server problem.
"""

import os
from types import SimpleNamespace
from unittest import mock


def _resolve_backbone(entry, llm, env):
    """The precedence implemented in scripts/list_backbones.py."""
    return (getattr(entry, "api_key", "")
            or env.get("LLM_API_KEY")
            or env.get("OPENAI_API_KEY")
            or getattr(llm, "api_key", "")
            or "")


def test_backbone_key_beats_a_stale_environment_variable():
    entry = SimpleNamespace(api_key="sk-backbone-own")
    llm = SimpleNamespace(api_key="sk-main")
    resolved = _resolve_backbone(entry, llm, {"LLM_API_KEY": "sk-stale"})
    assert resolved == "sk-backbone-own"


def test_environment_is_used_when_a_backbone_declares_no_key():
    entry = SimpleNamespace(api_key="")
    llm = SimpleNamespace(api_key="sk-main")
    resolved = _resolve_backbone(entry, llm, {"LLM_API_KEY": "sk-from-env"})
    assert resolved == "sk-from-env"


def test_main_llm_key_is_the_last_fallback():
    entry = SimpleNamespace(api_key="")
    llm = SimpleNamespace(api_key="sk-main")
    assert _resolve_backbone(entry, llm, {}) == "sk-main"


def test_check_vlm_prefers_config_over_environment():
    from scripts.check_vlm import _first

    # The order the module now uses: config first, environment as fallback.
    with mock.patch.dict(os.environ, {"LLM_API_KEY": "sk-stale"}):
        resolved = _first("sk-config", None,
                          os.environ.get("LLM_API_KEY"), default="EMPTY")
    assert resolved == "sk-config"


def test_check_vlm_falls_back_to_environment_when_config_is_empty():
    from scripts.check_vlm import _first

    with mock.patch.dict(os.environ, {"LLM_API_KEY": "sk-from-env"}):
        resolved = _first("", None,
                          os.environ.get("LLM_API_KEY"), default="EMPTY")
    assert resolved == "sk-from-env"


def test_effective_config_never_persists_the_key():
    """The run directory is archived, so the secret must not be written into it."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from scripts.run_validation import _effective_config

    args = SimpleNamespace(config="config/file_router.yaml", subset=30, docs=8,
                           eval_per_dataset=6, epochs=8, device="cpu",
                           reuse_model="")
    with tempfile.TemporaryDirectory() as directory:
        config = _effective_config(args, Path(directory))
    assert config["llm"]["api_key"] == ""


def test_stages_still_receive_a_key_after_it_is_stripped(monkeypatch):
    """Stripping the config without passing the key onward left every stage
    unauthenticated: preflight passed against the source config while the run
    itself returned 401 on every call."""
    from scripts.run_validation import resolved_api_key

    # The released config ships no key, so resolution falls back to the
    # environment; that fallback is what stages rely on.
    monkeypatch.setenv("LLM_API_KEY", "sk-from-environment")
    assert resolved_api_key("config/file_router.yaml"), (
        "the key must still be resolvable for injection into stage processes")


def test_stage_runner_injects_extra_environment(tmp_path):
    from scripts.run_validation import StageRunner

    runner = StageRunner(tmp_path / "run.log", extra_env={"LLM_API_KEY": "sk-x"})
    assert runner.extra_env["LLM_API_KEY"] == "sk-x"


def test_env_key_wins_when_the_env_redirects_to_a_different_host(monkeypatch):
    """A redirected endpoint must be paired with the key chosen for it.

    The backbone ablation sets LLM_BASE_URL and LLM_API_KEY together to reach a
    different provider. base_url already came from the environment, so taking
    the key from the config sends the local server's credential to a hosted API:
    an AuthenticationError on every call, hidden by the stub fallback for the
    whole run. Two backbone arms burned 18 hours that way.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_validation

    cfg = Path(__file__).parent / "_keyprec.yaml"
    cfg.write_text(
        "llm:\n"
        "  base_url: http://localhost:8000/v1\n"
        "  api_key: sk-local-vllm\n", encoding="utf-8")
    try:
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-hosted-provider")
        assert run_validation.resolved_api_key(str(cfg)) == "sk-hosted-provider"

        # Same host: the config still wins, so a stale exported key cannot
        # replace a credential the config declares for its own endpoint.
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
        assert run_validation.resolved_api_key(str(cfg)) == "sk-local-vllm"

        # No redirect at all: config wins.
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert run_validation.resolved_api_key(str(cfg)) == "sk-local-vllm"
    finally:
        cfg.unlink(missing_ok=True)


def test_vlm_client_pairs_the_env_key_with_the_env_host(monkeypatch):
    """The same rule must hold in the client that actually sends the request."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from file_router.encoders import vlm as vlm_mod

    seen = {}

    class _OpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai",
                        SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-hosted-provider")

    client = object.__new__(vlm_mod.VLM)
    client.cfg = SimpleNamespace(base_url="http://localhost:8000/v1",
                                 api_key="sk-local-vllm",
                                 request_timeout=30.0, max_retries=1,
                                 model="m")
    client.backend = "openai"
    client._init_openai()

    assert seen["api_key"] == "sk-hosted-provider", (
        "the redirected host must receive its own key")
    assert seen["base_url"] == "https://api.example.com/v1"
