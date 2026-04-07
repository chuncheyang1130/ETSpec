import random
from datasets import load_dataset

# Keep the same fluent QUERY_TEMPLATE as before
QUERY_TEMPLATE = """
Provide your step-by-step reasoning. On the last line by itself, give the final answer in the format "Answer: <LETTER>".
Question: {question}
Options:
{options}

Answer:
""".strip()

def load_mmlu_pro_dataset(sample_per_category: int = 20, seed: int = 0):
    """
    Returns a list of formatted MMLU‑Pro prompts, sampling up to `sample_per_category`
    examples from each category.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # group by category
    by_cat = {}
    for ex in ds:
        cat = ex["category"]
        by_cat.setdefault(cat, []).append(ex)
    # reproducible sampling
    rng = random.Random(seed)
    prompts = []
    for cat, examples in by_cat.items():
        picks = rng.sample(examples, min(sample_per_category, len(examples)))
        for ex in picks:
            opts = ex["options"]
            options_str = "\n".join(
                f"({chr(ord('A')+i)}) {opt}"
                for i, opt in enumerate(opts)
            )
            prompt = QUERY_TEMPLATE.format(
                question=ex["question"],
                options=options_str
            )
            prompts.append(prompt)
    return prompts

def load_mmlu_pro_dataset_answer(sample_per_category: int = 20, seed: int = 42):
    """
    Returns a list of dicts {"question": prompt, "answer": <"A"–"J">},
    sampling up to `sample_per_category` examples per category.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_cat = {}
    for ex in ds:
        by_cat.setdefault(ex["category"], []).append(ex)
    rng = random.Random(seed)
    examples = []
    for cat, items in by_cat.items():
        picks = rng.sample(items, min(sample_per_category, len(items)))
        for ex in picks:
            opts = ex["options"]
            options_str = "\n".join(
                f"({chr(ord('A')+i)}) {opt}"
                for i, opt in enumerate(opts)
            )
            prompt = QUERY_TEMPLATE.format(
                question=ex["question"],
                options=options_str
            )
            answer_letter = chr(ord('A') + ex["answer_index"])
            examples.append({"question": prompt, "answer": answer_letter})
    return examples
