import base64
import json
import pickle
import zlib

import pytest

from run.pipelines.benchmarks.loader import livecodebench


def _entry(**overrides):
    entry = {
        "question_id": "q1",
        "contest_date": "2025-02-01T00:00:00",
        "platform": "leetcode",
        "difficulty": "easy",
        "question_content": "Echo the input integer.",
        "starter_code": "class Solution:\n    def echo(self, value: int) -> int:",
        "public_test_cases": '[{"input": "1\\n", "output": "1\\n"}]',
        "private_test_cases": base64.b64encode(
            zlib.compress(pickle.dumps('[{"input": "2\\n", "output": "2\\n"}]'))
        ).decode("utf-8"),
        "metadata": json.dumps({"func_name": "echo"}),
    }
    entry.update(overrides)
    return entry


def _fake_release(monkeypatch, entries):
    """Stub the hub fetch so tests never touch the network."""

    def fake_iter(version_tag):
        assert version_tag == livecodebench.DEFAULT_VERSION_TAG
        return iter(entries)

    monkeypatch.setattr(livecodebench, "_iter_release_entries", fake_iter)


def test_livecodebench_loader_matches_code_benchmark_contract(monkeypatch):
    _fake_release(monkeypatch, [_entry()])
    samples = livecodebench.load_livecodebench_dataset(query_version="qwen")

    assert len(samples) == 1
    assert "Echo the input integer." in samples[0]["query"]
    assert "class Solution:" in samples[0]["query"]
    assert set(samples[0]) == {
        "query",
        "testcase",
        "question_id",
        "contest_date",
        "platform",
        "difficulty",
    }
    assert samples[0]["testcase"] == {
        "inputs": ["1\n", "2\n"],
        "outputs": ["1\n", "2\n"],
        "fn_name": "echo",
    }


def test_livecodebench_loader_uses_stdio_format_without_starter_code(monkeypatch):
    _fake_release(
        monkeypatch,
        [
            _entry(
                question_content="Echo stdin.",
                starter_code="",
                public_test_cases=[],
                private_test_cases=[
                    {"input": "3\n", "output": "3\n", "testtype": "stdin"}
                ],
                metadata={},
            )
        ],
    )

    sample = livecodebench.load_livecodebench_dataset()[0]

    assert "Read the inputs from stdin" in sample["query"]
    assert sample["testcase"]["fn_name"] is None


def test_livecodebench_loader_rejects_unknown_query_version():
    with pytest.raises(ValueError, match="Unknown query_version"):
        livecodebench.load_livecodebench_dataset(query_version="unknown")


def test_default_window_is_january_through_april_2025():
    assert livecodebench.DEFAULT_START_DATE == "2025-01-01"
    assert livecodebench.DEFAULT_END_DATE == "2025-05-01"


def test_window_is_half_open_on_both_bounds(monkeypatch):
    entries = [
        _entry(question_id="before", contest_date="2024-12-31T23:59:59"),
        _entry(question_id="lower-bound", contest_date="2025-01-01T00:00:00"),
        _entry(question_id="inside", contest_date="2025-04-06T00:00:00"),
        _entry(question_id="upper-bound", contest_date="2025-05-01T00:00:00"),
    ]
    _fake_release(monkeypatch, entries)

    samples = livecodebench.load_livecodebench_dataset()

    assert [s["question_id"] for s in samples] == ["lower-bound", "inside"]


def test_selection_is_sorted_by_contest_date(monkeypatch):
    _fake_release(
        monkeypatch,
        [
            _entry(question_id="march", contest_date="2025-03-01T00:00:00"),
            _entry(question_id="january", contest_date="2025-01-15T00:00:00"),
            _entry(question_id="february", contest_date="2025-02-01T00:00:00"),
        ],
    )

    samples = livecodebench.load_livecodebench_dataset()

    assert [s["question_id"] for s in samples] == ["january", "february", "march"]


def test_expected_count_mismatch_is_rejected(monkeypatch):
    _fake_release(monkeypatch, [_entry()])

    with pytest.raises(ValueError, match="expected 2"):
        livecodebench.load_livecodebench_dataset(expected_count=2)


def test_unknown_version_tag_is_rejected():
    with pytest.raises(ValueError, match="Unknown version_tag"):
        livecodebench.load_livecodebench_dataset(version_tag="v99")


def test_start_date_must_precede_end_date():
    with pytest.raises(ValueError, match="must be strictly before"):
        livecodebench.load_livecodebench_dataset(
            start_date="2025-05-01", end_date="2025-01-01"
        )
