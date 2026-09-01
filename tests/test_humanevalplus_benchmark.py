from run.pipelines.benchmarks import registry
from run.pipelines.benchmarks.evaluation.code_eval.utils import build_test_code
from run.pipelines.benchmarks.loader import humanevalplus


def test_human_eval_name_resolves_to_humanevalplus():
    assert "human-eval" in registry.AVAILABLE_BENCHMARKS
    assert "humanevalplus" not in registry.AVAILABLE_BENCHMARKS
    assert registry._LOADER_CONFIG["human-eval"] == (
        ".humanevalplus",
        "load_humanevalplus_dataset",
    )


def test_humanevalplus_loader_uses_augmented_tests(monkeypatch):
    augmented_test = "def check(candidate):\n    assert candidate(2) == 3"

    def fake_load_dataset(name, split):
        assert name == "evalplus/humanevalplus"
        assert split == "test"
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "def successor(x):\n    pass",
                "test": augmented_test,
                "entry_point": "successor",
            }
        ]

    monkeypatch.setattr(humanevalplus, "load_dataset", fake_load_dataset)
    samples = humanevalplus.load_humanevalplus_dataset(query_version="qwen")

    assert len(samples) == 1
    assert samples[0]["task_id"] == "HumanEval/0"
    assert "def successor" in samples[0]["query"]
    assert samples[0]["testcase"] == augmented_test
    assert samples[0]["entry_point"] == "successor"
    assert samples[0]["test_timeout"] == 10.0


def test_human_eval_builds_humanevalplus_test_program():
    program = build_test_code(
        "def successor(x):\n    return x + 1",
        "def check(candidate):\n    assert candidate(2) == 3",
        "successor",
        "human-eval",
    )

    assert program.endswith("check(successor)")
