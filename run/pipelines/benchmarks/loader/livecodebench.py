# LIVECODEBENCH_QUERY_TEMPLATE is adapted from https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
from datasets import load_dataset
import json

QUERY_TEMPLATE = """
You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Question: {Question}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

```python\n# YOUR CODE HERE\n```
""".strip()

# LIVECODEBENCH
def load_livecodebench_dataset():

    dataset = load_dataset("livecodebench/code_generation_lite", "v4_v5", trust_remote_code=True) # problems released between Aug 2024 and Jan 2025. The deepseek eval dataset setting
    formatted_dataset = [QUERY_TEMPLATE.format(Question=entry['question_content']) for entry in dataset['test']]

    return formatted_dataset

# WARNING: This function is NOT ready
def load_livecodebench_dataset_answer():
    """
    Returns a list of dicts like:
    {
    "question": <formatted prompt>,
    "meta": <the raw row (incl. tests, difficulty, platform, etc.)>
    }
    which is what run_livecodebench_eval above expects.
    """
    ds = load_dataset("livecodebench/code_generation_lite", "v4_v5", trust_remote_code=True)['test']
    examples = []
    for row in ds:
        q = QUERY_TEMPLATE.format(Question=row["question_content"])
        # keep raw row to access fields: public/private tests, platform, difficulty, contest_date, etc.
        meta = {
            "question_id": row.get("question_id"),
            "platform": row.get("platform"),
            "difficulty": row.get("difficulty"),
            "contest_date": row.get("contest_date"),
            "public_test_cases": row.get("public_test_cases"),
            "private_test_cases": row.get("private_test_cases"),
            "meta_data": row.get("meta_data", {}),
        }
        examples.append({
            "question": q,
            "meta": meta
        })
    return examples


# def load_livecodebench_dataset_answer():
#     """
#     Returns a list of dicts, each containing:
#     - 'prompt' : the templated question string
#     - 'public_tests' : JSON string of public testcases
#     - 'private_tests': JSON string of hidden testcases
#     - 'starter_code' : any provided starter code string (may be empty)
#     - 'difficulty' : problem difficulty tag
#     - 'metadata' : extra metadata
#     """
#     ds = load_dataset(
#     "livecodebench/code_generation_lite",
#     split="test",
#     version_tag="v4_v5",
#     trust_remote_code=True
#     )
#     examples = []
#     for ex in ds:
#         examples.append({
#             "prompt": QUERY_TEMPLATE.format(Question=ex["question_content"]),
#             "public_tests" : ex["public_test_cases"],
#             "private_tests": ex["private_test_cases"],
#             "starter_code" : ex["starter_code"],
#             "difficulty"   : ex["difficulty"],
#             "metadata"     : ex["metadata"],
#         })
#     return examples