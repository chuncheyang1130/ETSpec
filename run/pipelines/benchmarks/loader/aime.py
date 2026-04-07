# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}
""".strip()

# AIME 2024
def load_aime_dataset():
    """
    Returns list of formatted question strings for AIME‑2024 (30 problems).
    """
    # ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    ds = load_dataset("yentinglin/aime_2025", split="train")
    return [
        QUERY_TEMPLATE.format(Question=example["problem"])
        for example in ds
    ]

def load_aime_dataset_answer():
    """
    Returns list of dicts with 'question' and 'answer' for AIME‑2024.
    """
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    examples = []
    for entry in ds:
        q = QUERY_TEMPLATE.format(Question=entry["problem"])
        a = entry["answer"]  # the numeric answer string :contentReference[oaicite:0]{index=0}
        examples.append({"question": q, "answer": a})
    return examples
