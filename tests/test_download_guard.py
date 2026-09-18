"""A smaller run must not silently destroy a larger corpus.

download_data.py overwrites data/raw, and check_data_scale runs afterwards, so
a smoke run used to shrink a medium corpus and only report it once the data was
already gone.  This happened three times during development.
"""

from unittest import mock

from scripts.download_data import _would_downgrade


def _with_existing(count):
    return mock.patch("scripts.download_data._existing_qa_count",
                      return_value=count)


def test_smoke_over_medium_is_blocked():
    with _with_existing(300):
        assert _would_downgrade(None, ["mmdocrag"], 30) == [
            ("mmdocrag", 300, 30)]


def test_same_scale_is_allowed():
    with _with_existing(300):
        assert _would_downgrade(None, ["mmdocrag"], 300) == []


def test_larger_scale_is_allowed():
    with _with_existing(300):
        assert _would_downgrade(None, ["mmdocrag"], 1000) == []


def test_download_everything_is_never_a_downgrade():
    """subset_qa=0 means no truncation, so it cannot shrink anything."""
    with _with_existing(300):
        assert _would_downgrade(None, ["mmdocrag"], 0) == []


def test_empty_disk_is_never_a_downgrade():
    with _with_existing(0):
        assert _would_downgrade(None, ["mmdocrag"], 30) == []


def test_small_differences_do_not_trip_the_guard():
    """Per-dataset source caps mean counts never match the request exactly."""
    with _with_existing(310):
        assert _would_downgrade(None, ["mmdocrag"], 300) == []
