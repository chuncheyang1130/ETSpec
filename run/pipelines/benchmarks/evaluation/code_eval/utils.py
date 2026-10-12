import subprocess
import re
import os
import tempfile
import json
import sys
from decimal import Decimal, InvalidOperation

def extract_code(text: str) -> str:
    #Extracts code from a ```python ... ``` block.
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.S)
    if match:
        return match.group(1).strip()
    
    # Pattern to match ```...``` blocks
    pattern = r"```(?:\w+)?\n?(.*?)\n?```"
    # (+ ```) as we add the opening "```python" to the gen_prefix
    matches = re.findall(pattern, r"```" + text, re.DOTALL)
    # if no matches, try to match ```...``` blocks (after removing the language)
    if not matches:
        text_without_lang = re.sub(r"```python", "```", text)
        matches = re.findall(pattern, text_without_lang, re.DOTALL)
    if not matches:
        return ""
    else:
        return matches[0]

def build_test_code(code_str: str, testcase_str: str, entry_point: str = None, bench_name: str = "humaneval") -> str:
    """Builds a complete code string that includes the solution and test cases."""
    test_code = ""
    if bench_name == "human-eval":
        test_code = f"{code_str}\n\n{testcase_str}\ncheck({entry_point})"
    elif bench_name == "mbpp":
        test_code = f"{code_str}\n\n{testcase_str}"
    else:
        raise NotImplementedError(f"Test code generation not implemented for benchmark: {bench_name}")
    
    return test_code
    
def run_single_test(code_str: str, timeout: float = 2.0) -> bool:
    """Run the generated code with the provided test cases and return whether all tests passed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_path = os.path.join(temp_dir, "solution.py")
        
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code_str)

        try:
            proc = subprocess.run(
                ["python", code_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            # If returncode is 0 and no assertion errors, tests passed
            if proc.returncode == 0:
                return True, None
            else:
                # Check if it's an assertion error
                # if "AssertionError" in proc.stderr or "assert" in proc.stderr.lower():
                #     return False
                # Other runtime errors also count as failure
                return False, proc.stderr
        except (subprocess.TimeoutExpired, Exception) as e:
            return False, str(e)


_LIVECODEBENCH_IMPORTS = """\
import sys
from bisect import *
from collections import *
from copy import *
from functools import *
from heapq import *
from itertools import *
from math import *
from statistics import *
from typing import *
"""


def _stdio_output_matches(actual: str, expected: str) -> bool:
    actual_lines = [line.strip() for line in actual.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if actual_lines == expected_lines:
        return True
    if len(actual_lines) != len(expected_lines):
        return False

    try:
        return all(
            [Decimal(token) for token in actual_line.split()]
            == [Decimal(token) for token in expected_line.split()]
            for actual_line, expected_line in zip(actual_lines, expected_lines)
        )
    except InvalidOperation:
        return False


def _build_call_test_code(
    code_str: str,
    fn_name: str,
    testcase_input: str,
    expected_output: str,
) -> str:
    arguments = [json.loads(line) for line in testcase_input.splitlines()]
    expected = json.loads(expected_output)
    return (
        f"{_LIVECODEBENCH_IMPORTS}\n{code_str}\n\n"
        "__lcb_owner = Solution() if 'Solution' in globals() else __import__('sys').modules[__name__]\n"
        f"__lcb_result = getattr(__lcb_owner, {fn_name!r})(*{arguments!r})\n"
        "if isinstance(__lcb_result, tuple):\n"
        "    __lcb_result = list(__lcb_result)\n"
        f"__lcb_expected = {expected!r}\n"
        "assert __lcb_result == __lcb_expected, "
        "f'expected {__lcb_expected!r}, got {__lcb_result!r}'\n"
        "print('__LIVECODEBENCH_TEST_PASSED__')\n"
    )


def run_livecodebench_tests(
    code_str: str,
    testcase: dict,
    timeout: float = 2.0,
):
    """Run normalized LiveCodeBench stdin or call-based test cases."""
    inputs = testcase.get("inputs", [])
    outputs = testcase.get("outputs", [])
    fn_name = testcase.get("fn_name")
    if not inputs or len(inputs) != len(outputs):
        return False, "Invalid or empty LiveCodeBench testcase"

    for testcase_input, expected_output in zip(inputs, outputs):
        try:
            test_code = (
                _build_call_test_code(
                    code_str,
                    fn_name,
                    testcase_input,
                    expected_output,
                )
                if fn_name
                else code_str
            )
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return False, f"Invalid functional testcase: {error}"

        with tempfile.TemporaryDirectory() as temp_dir:
            code_path = os.path.join(temp_dir, "solution.py")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(test_code)

            try:
                proc = subprocess.run(
                    [sys.executable, code_path],
                    input=None if fn_name else testcase_input,
                    capture_output=True,
                    timeout=timeout,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                return False, "Time Limit Exceeded"
            except Exception as error:
                return False, str(error)

        if proc.returncode != 0:
            return False, proc.stderr.strip() or "Runtime Error"
        if fn_name and (
            not proc.stdout.splitlines()
            or proc.stdout.splitlines()[-1] != "__LIVECODEBENCH_TEST_PASSED__"
        ):
            return False, "Functional test did not complete"
        if not fn_name and not _stdio_output_matches(proc.stdout, expected_output):
            return False, (
                f"Wrong Answer: expected {expected_output.strip()!r}, "
                f"got {proc.stdout.strip()!r}"
            )

    return True, None
