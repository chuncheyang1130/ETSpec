# LiveCodeBench code-generation loader.
#
# Two things make the obvious `load_dataset(...)` call fail on this repo:
#
# 1. `livecodebench/code_generation` only ever contained `test.jsonl`
#    (release_v1, 400 problems, May 2023 - Mar 2024). Asking it for
#    `release_v5` cannot work. The maintained repo is
#    `livecodebench/code_generation_lite`.
# 2. `code_generation_lite` ships a *loading script*, and `datasets >= 3`
#    removed script execution entirely (`trust_remote_code` is gone), so it
#    cannot be loaded through `load_dataset` on datasets 4.x at all.
#
# Both are avoided by fetching the plain per-release JSONL increments straight
# from the hub and parsing them here. That keeps the official data source and
# reproduces the official release semantics without needing script support.

import base64
import json
import pickle
import zlib
from datetime import datetime

from huggingface_hub import hf_hub_download


_REPO_ID = "livecodebench/code_generation_lite"

# Each `testN.jsonl` is the *increment* added by release vN, matching the
# ALLOWED_FILES table in the upstream loading script. A release tag is the
# cumulative union; a `vA_vB` tag is the half-open span of increments.
_RELEASE_FILES = {
    "v1": ["test.jsonl"],
    "v2": ["test2.jsonl"],
    "v3": ["test3.jsonl"],
    "v4": ["test4.jsonl"],
    "v5": ["test5.jsonl"],
    "v6": ["test6.jsonl"],
    "v4_v5": ["test4.jsonl", "test5.jsonl"],
    "v5_v6": ["test5.jsonl", "test6.jsonl"],
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl",
        "test6.jsonl",
    ],
}

# The evaluation window: January through April 2025, as a half-open range.
# LiveCodeBench's newest problem is 2025-04-06 (v6 is the final release; the
# dataset has not been updated since 2025-06), so this window is also the whole
# 2025 portion of the benchmark: 182 questions (51 Jan / 51 Feb / 68 Mar / 12 Apr).
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2025-05-01"

# `v5_v6` is the smallest pair of increments that fully contains the window
# above. The increments are chronological and contiguous -- v5 spans
# 2024-09-22..2025-01-04 and v6 spans 2025-01-04..2025-04-06 -- so every 2025
# problem lives in one of the two, and no earlier increment can contribute.
# Widen this if you move `start_date` before 2024-09-22.
DEFAULT_VERSION_TAG = "v5_v6"


QWEN_QUERY_TEMPLATE = r"""
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {Question}

{Format}
""".strip()

LLAMA_QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE

STDIN_FORMAT = r"""
Read the inputs from stdin, solve the problem, and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the Python program runs, it reads the inputs, runs the algorithm, and writes output to STDOUT.

```python
# YOUR CODE HERE
```
""".strip()

STARTER_CODE_FORMAT = r"""
Use the following starter code to write the solution to the problem and enclose your complete code within delimiters.

```python
{StarterCode}
```
""".strip()


def _decode_test_cases(field):
    """Decode LiveCodeBench's JSON or base64/zlib/pickle test representation."""
    if isinstance(field, list):
        return field
    if not isinstance(field, (str, bytes)):
        raise ValueError(f"Unsupported test-case type: {type(field).__name__}")

    if isinstance(field, bytes):
        field = field.decode("utf-8")

    try:
        decoded = json.loads(field)
    except json.JSONDecodeError:
        decoded = pickle.loads(zlib.decompress(base64.b64decode(field)))
        if isinstance(decoded, bytes):
            decoded = decoded.decode("utf-8")
        if isinstance(decoded, str):
            decoded = json.loads(decoded)

    if not isinstance(decoded, list):
        raise ValueError("Decoded LiveCodeBench test cases must be a list")
    return decoded


def _decode_metadata(field):
    if isinstance(field, dict):
        return field
    if not field:
        return {}
    return json.loads(field)


def _build_testcase(entry):
    tests = [
        *_decode_test_cases(entry["public_test_cases"]),
        *_decode_test_cases(entry["private_test_cases"]),
    ]
    metadata = _decode_metadata(entry.get("metadata", {}))

    return {
        "inputs": [test["input"] for test in tests],
        "outputs": [test["output"] for test in tests],
        "fn_name": metadata.get("func_name"),
    }


def _parse_date(value):
    """Parse a `contest_date` or a plain `YYYY-MM-DD` bound."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    # contest_date is ISO like '2025-01-04T18:30:00'; bounds are 'YYYY-MM-DD'.
    return datetime.fromisoformat(text.replace("Z", "+00:00").split("+")[0])


def _iter_release_entries(version_tag):
    files = _RELEASE_FILES.get(version_tag)
    if files is None:
        raise ValueError(
            f"Unknown version_tag {version_tag!r}; "
            f"expected one of {sorted(_RELEASE_FILES)}"
        )
    for name in files:
        path = hf_hub_download(
            repo_id=_REPO_ID, filename=name, repo_type="dataset"
        )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_livecodebench_dataset(
    query_version: str = "llama",
    version_tag: str = DEFAULT_VERSION_TAG,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    expected_count: int | None = None,
):
    """Load LiveCodeBench problems whose contest date falls in a window.

    The window is half-open, ``start_date <= contest_date < end_date``, so month
    bounds compose without double-counting a boundary contest. Pass
    ``start_date=None`` and ``end_date=None`` to disable filtering and take the
    whole release.

    ``expected_count``, when given, asserts the number of problems selected --
    useful for pinning an evaluation window so a silent upstream change to the
    dataset does not quietly alter the benchmark.
    """
    if query_version == "qwen":
        query_template = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        query_template = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")

    lower = _parse_date(start_date) if start_date else None
    upper = _parse_date(end_date) if end_date else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f"start_date {start_date!r} must be strictly before end_date {end_date!r}"
        )

    selected = []
    for entry in _iter_release_entries(version_tag):
        contest_date = entry.get("contest_date")
        if (lower is not None or upper is not None) and not contest_date:
            raise ValueError(
                "LiveCodeBench entry has no contest_date; cannot filter by window"
            )
        if contest_date:
            moment = _parse_date(contest_date)
            if lower is not None and moment < lower:
                continue
            if upper is not None and moment >= upper:
                continue
        selected.append(entry)

    # Deterministic order regardless of file order, so a run is reproducible
    # before the registry applies its own seeded shuffle.
    selected.sort(
        key=lambda e: (str(e.get("contest_date", "")), str(e.get("question_id", "")))
    )

    if expected_count is not None and len(selected) != int(expected_count):
        raise ValueError(
            f"LiveCodeBench window [{start_date}, {end_date}) on {version_tag} "
            f"selected {len(selected)} problems, expected {expected_count}"
        )

    samples = []
    for entry in selected:
        starter_code = entry.get("starter_code", "")
        format_instruction = (
            STARTER_CODE_FORMAT.format(StarterCode=starter_code)
            if starter_code
            else STDIN_FORMAT
        )
        samples.append(
            {
                "query": query_template.format(
                    Question=entry["question_content"],
                    Format=format_instruction,
                ),
                "testcase": _build_testcase(entry),
                "question_id": entry.get("question_id"),
                "contest_date": entry.get("contest_date"),
                "platform": entry.get("platform"),
                "difficulty": entry.get("difficulty"),
            }
        )

    return samples
