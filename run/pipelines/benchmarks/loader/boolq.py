from datasets import load_dataset as _hf

def load_boolq_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("super_glue", "boolq", split="validation"):
        samples.append({
            "context": f"{entry['passage']}\nQuestion: {entry['question']}?\nAnswer:",
            "choices": [" no", " yes"],
            "answer": int(entry["label"]),
        })
    return samples
