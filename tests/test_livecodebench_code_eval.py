from run.pipelines.benchmarks.evaluation.code_eval import run_livecodebench_tests


def test_run_livecodebench_stdio_tests():
    testcase = {
        "inputs": ["2\n", "7\n"],
        "outputs": ["4\n", "14\n"],
        "fn_name": None,
    }

    passed, status = run_livecodebench_tests(
        "value = int(input())\nprint(value * 2)", testcase
    )

    assert passed is True
    assert status is None


def test_run_livecodebench_functional_tests():
    testcase = {
        "inputs": ["[1, 2, 3]\n2", "[4, 5]\n3"],
        "outputs": ["[2, 4, 6]", "[12, 15]"],
        "fn_name": "multiply",
    }
    solution = """
class Solution:
    def multiply(self, values: List[int], factor: int) -> List[int]:
        return [value * factor for value in values]
"""

    passed, status = run_livecodebench_tests(solution, testcase)

    assert passed is True
    assert status is None
