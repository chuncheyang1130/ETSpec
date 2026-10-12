from datasets import load_dataset as _hf

def load_openbookqa_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("allenai/openbookqa", "main", split="test"):
        labels = entry["choices"]["label"]
        samples.append({
            "context": entry["question_stem"],
            "choices": [f" {t}" for t in entry["choices"]["text"]],
            "answer": labels.index(entry["answerKey"]),
        })
    return samples
