"""CLIP feature extraction must survive the transformers 4.x -> 5.x change.

transformers <5 returned a tensor from get_text_features / get_image_features;
5.x returns a BaseModelOutputWithPooling. Indexing that object yields
last_hidden_state -- a token sequence of a different width -- so the old code
silently produced 2048-d query vectors against a 512-d index and disabled visual
retrieval for entire campaigns while the runs still completed.
"""

import pytest

torch = pytest.importorskip("torch")

from file_router.encoders.visual_encoder import _as_embedding


def test_plain_tensor_passes_through():
    """The transformers 4.x return shape."""
    assert tuple(_as_embedding(torch.randn(1, 512)).shape) == (1, 512)


def test_text_embeds_attribute_is_preferred_over_hidden_states():
    """The 5.x shape: the projected embedding, not the token sequence."""

    class Output:
        text_embeds = torch.randn(1, 512)
        last_hidden_state = torch.randn(1, 7, 2048)

    assert tuple(_as_embedding(Output()).shape) == (1, 512)


def test_image_embeds_attribute_is_handled():
    class Output:
        image_embeds = torch.randn(1, 512)

    assert tuple(_as_embedding(Output()).shape) == (1, 512)


def test_pooler_output_is_used_when_embeds_are_absent():
    class Output:
        pooler_output = torch.randn(1, 512)

    assert tuple(_as_embedding(Output()).shape) == (1, 512)


def test_token_sequence_is_pooled_not_truncated():
    """A 3-D tensor must be mean-pooled so it stays comparable to the index."""
    assert tuple(_as_embedding(torch.randn(1, 7, 512)).shape) == (1, 512)


def test_unrecognised_structure_raises_rather_than_guessing():
    """Silently returning the wrong field is what caused the original bug."""

    class Output:
        pass

    with pytest.raises(TypeError):
        _as_embedding(Output())
