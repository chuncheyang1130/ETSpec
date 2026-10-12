import re
from datasets import load_dataset as _hf

def _preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")

def load_hellaswag_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("Rowan/hellaswag", split="validation"):
        ctx = entry["ctx_a"] + " " + entry["ctx_b"].capitalize()
        samples.append({
            "context": _preprocess(entry["activity_label"] + ": " + ctx),
            "choices": [" " + _preprocess(e) for e in entry["endings"]],
            "answer": int(entry["label"]),
        })
    return samples
