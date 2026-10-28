from datasets import load_dataset as _hf

def load_rte_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("super_glue", "rte", split="validation"):
        samples.append({
            "context": f"{entry['premise']}\nQuestion: {entry['hypothesis']} True or False?\nAnswer:",
            "choices": [" True", " False"],
            "answer": int(entry["label"]),
        })
    return samples
