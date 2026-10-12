from datasets import load_dataset as _hf

# REAP's MC suite is scored zero-shot by log-likelihood (lm-eval-harness protocol),
# so loaders return the raw context + the choice strings, NOT a generation prompt.
# ARC reports byte-length-normalised accuracy (acc_norm) in the REAP tables.

def _load_arc(subset: str):
    samples = []
    for entry in _hf("allenai/ai2_arc", subset, split="test"):
        labels = entry["choices"]["label"]
        texts = entry["choices"]["text"]
        if entry["answerKey"] not in labels:
            continue
        samples.append({
            "context": f"Question: {entry['question']}\nAnswer:",
            "choices": [f" {t}" for t in texts],
            "answer": labels.index(entry["answerKey"]),
        })
    return samples

def load_arc_challenge_dataset(query_version: str = "llama"):
    return _load_arc("ARC-Challenge")

def load_arc_easy_dataset(query_version: str = "llama"):
    return _load_arc("ARC-Easy")
