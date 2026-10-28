from datasets import load_dataset as _hf

# WinoGrande scores the SHARED suffix after the blank, conditioned on each
# candidate filling, so context varies per choice. We emit per-choice contexts.

def load_winogrande_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("allenai/winogrande", "winogrande_xl", split="validation"):
        idx = entry["sentence"].index("_")
        suffix = entry["sentence"][idx + 1:].strip()
        samples.append({
            "contexts": [entry["sentence"][:idx] + entry[f"option{i}"] for i in (1, 2)],
            "choices": [f" {suffix}", f" {suffix}"],
            "answer": int(entry["answer"]) - 1,
        })
    return samples
