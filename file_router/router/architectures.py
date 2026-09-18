"""Router network architectures, defined once for training and inference.

The network was previously built inline in both trainer.py and scorer.py. Two
copies of the same literal is how a trained checkpoint stops loading: change the
shape in one place and the other raises a size-mismatch at load time, which the
scorer catches and reports as "falling back" -- a silent downgrade to the
untrained path. Everything that builds a router goes through build() here, and
the shape is recorded in router_meta.json so inference reconstructs exactly what
training produced.
"""

from __future__ import annotations

from typing import List


def hidden_sizes(arch: str, hidden_dim: int) -> List[int]:
    """Hidden layer widths for a named architecture.

    `mlp2` is the shipped default and must keep its exact widths so existing
    checkpoints stay loadable.
    """
    if arch == "linear":
        return []
    if arch == "mlp1":
        return [hidden_dim // 4]
    if arch == "mlp2":
        return [hidden_dim, hidden_dim // 2]
    if arch == "mlp3":
        return [hidden_dim * 2, hidden_dim, hidden_dim // 2]
    raise ValueError(
        f"unknown router architecture {arch!r}; "
        f"expected one of linear, mlp1, mlp2, mlp3")


def build(input_dim: int, hidden_dim: int, arch: str = "mlp2"):
    """Construct the scoring network for `arch`."""
    import torch.nn as nn

    widths = hidden_sizes(arch, hidden_dim)
    layers, d = [], input_dim
    for w in widths:
        layers += [nn.Linear(d, w), nn.ReLU()]
        d = w
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def parameter_count(input_dim: int, hidden_dim: int, arch: str = "mlp2") -> int:
    """Trainable parameters, for reporting the capacity axis in the paper."""
    widths = hidden_sizes(arch, hidden_dim)
    total, d = 0, input_dim
    for w in widths:
        total += d * w + w
        d = w
    return total + d + 1
