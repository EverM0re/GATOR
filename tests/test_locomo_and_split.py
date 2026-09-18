from types import SimpleNamespace

from file_router.data.loaders import _locomo_session_pages
from file_router.data.split import choose_test_groups


def _cfg():
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            text=SimpleNamespace(chunk_chars=80),
            caption=SimpleNamespace(heuristic_max_chars=30),
        )
    )


def test_locomo_turn_evidence_maps_to_bounded_session_chunks():
    record = {
        "conversation": {
            "session_1_date_time": "1 Jan 2024",
            "session_1": [
                {"speaker": "A", "dia_id": "D1:01", "text": "first turn"},
                {"speaker": "B", "dia_id": "D1:2", "text": "x" * 80},
            ],
        },
        "session_summary": {"session_1_summary": "a useful session summary"},
    }
    pages, mapping = _locomo_session_pages(record, _cfg())
    assert len(pages) == 2
    assert mapping["D1:1"] == [0]
    assert mapping["D1:2"] == [1]
    assert all(len(page["caption"]) <= 30 for page in pages)


def test_group_split_is_deterministic_and_non_leaking():
    groups = ["doc-a", "doc-b", "doc-c", "doc-d"]
    first = choose_test_groups(groups, 0.25, 42)
    second = choose_test_groups(reversed(groups), 0.25, 42)
    assert first == second
    assert len(first) == 1
    assert first < set(groups)
