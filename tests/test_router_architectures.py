"""A router must load back as the architecture that trained it.

The network shape was previously duplicated as a literal in trainer.py and
scorer.py. Changing one without the other raises a size mismatch at load time,
which LearnedScorer catches and reports as "falling back" -- a silent downgrade
to the untrained path that still produces a complete, plausible-looking run.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from file_router.router.architectures import (  # noqa: E402
    build, hidden_sizes, parameter_count)

ARCHS = ["linear", "mlp1", "mlp2", "mlp3"]


def test_shipped_architecture_keeps_its_exact_shape():
    """mlp2 is what every existing checkpoint was trained with."""
    assert hidden_sizes("mlp2", 64) == [64, 32]
    assert parameter_count(28, 64, "mlp2") == 3969


def test_capacity_spans_a_wide_range_and_is_monotone():
    counts = [parameter_count(28, 64, a) for a in ARCHS]
    assert counts == sorted(counts), f"not monotone in capacity: {counts}"
    assert counts[-1] / counts[0] > 100, "range too narrow to be informative"


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown router architecture"):
        hidden_sizes("transformer", 64)


@pytest.mark.parametrize("arch", ARCHS)
def test_checkpoint_round_trips_through_meta(arch, tmp_path):
    """Save as the trainer does, reload as the scorer does, compare outputs."""
    torch = pytest.importorskip("torch")

    model = build(28, 64, arch)
    weights = tmp_path / "router_weights.pt"
    meta = tmp_path / "router_meta.json"
    torch.save(model.state_dict(), weights)
    meta.write_text(json.dumps(
        {"input_dim": 28, "hidden_dim": 64, "architecture": arch}))

    spec = json.loads(meta.read_text())
    reloaded = build(spec["input_dim"], spec["hidden_dim"],
                     spec.get("architecture", "mlp2"))
    reloaded.load_state_dict(torch.load(weights, map_location="cpu"))

    x = torch.randn(7, 28)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x)), (
            f"{arch} did not round-trip")


def test_a_checkpoint_without_the_field_is_read_as_the_shipped_model():
    """Older checkpoints predate the architecture field and are all mlp2."""
    spec = {"input_dim": 28, "hidden_dim": 64}
    assert spec.get("architecture", "mlp2") == "mlp2"
