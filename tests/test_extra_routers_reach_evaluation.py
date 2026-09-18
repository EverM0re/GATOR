"""EXTRA_ROUTERS must reach the list the evaluation loop actually iterates.

The opt-in arms were honoured everywhere that was easy to check -- the env var
parsed, the module-level ROUTERS grew, the startup banner printed -- while the
evaluation loop iterated `_available_routers()`, which was built from
BASE_ROUTERS alone. Three runs completed normally and produced the old six-arm
result, each costing ~16 minutes, before the mismatch was found.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _source_of(func_name):
    """Read from disk, not from an imported module.

    A module cached by an earlier import keeps the old source in memory, so
    inspect.getsource() on it cannot detect that the file regressed.
    """
    path = (Path(__file__).resolve().parents[1] / "scripts"
            / "evaluate_validation.py")
    text = path.read_text()
    start = text.index(f"def {func_name}(")
    nxt = text.find("\ndef ", start + 1)
    return text[start:nxt if nxt > 0 else len(text)]


def test_available_routers_appends_the_opt_in_arms(tmp_path, monkeypatch):
    """Check the returned tuple, not the source text.

    A comment mentioning EXTRA_ROUTERS satisfies a substring check even after
    the line that uses it is deleted, so assert on behaviour instead.
    """
    monkeypatch.setenv("EXTRA_ROUTERS", "bm25 dense cot selfask")
    sys.modules.pop("evaluate_validation", None)
    import evaluate_validation as ev

    cfg = SimpleNamespace(
        router=SimpleNamespace(router_model_path=""),
        paths=SimpleNamespace(router_model_dir=str(tmp_path)))
    names = ev._available_routers(cfg)
    for arm in ("bm25", "dense", "cot", "selfask"):
        assert arm in names, (
            f"{arm} never reaches the evaluation loop: _available_routers() "
            f"returned {names}")


def test_the_evaluation_loop_iterates_the_same_list_that_extra_arms_join():
    """Whatever the loop iterates must be the list EXTRA_ROUTERS extends."""
    import evaluate_validation as ev

    loop = inspect.getsource(ev.evaluate_qa)
    iterated = "_available_routers" in loop or "ROUTERS" in loop
    assert iterated, "could not identify the router list the loop iterates"

    if "_available_routers" in loop:
        assert "EXTRA_ROUTERS" in _source_of("_available_routers")


def test_unknown_arm_refuses_rather_than_running_the_old_set(monkeypatch):
    """A stale deployment must fail loudly, not reproduce the previous result."""
    monkeypatch.setenv("EXTRA_ROUTERS", "bm25 a_future_arm")
    for name in list(sys.modules):
        if name == "evaluate_validation":
            del sys.modules[name]
    try:
        import evaluate_validation  # noqa: F401
    except SystemExit as exc:
        assert "older than the caller" in str(exc)
    else:
        raise AssertionError("an unrecognised arm must abort the run")
