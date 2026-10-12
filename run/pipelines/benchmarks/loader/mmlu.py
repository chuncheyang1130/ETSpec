from datasets import load_dataset as _hf

_LETTERS = ["A", "B", "C", "D"]

def load_mmlu_dataset(query_version: str = "llama"):
    samples = []
    for entry in _hf("cais/mmlu", "all", split="test"):
        subject = entry["subject"].replace("_", " ")
        opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(entry["choices"]))
        samples.append({
            "context": (f"The following are multiple choice questions (with answers) "
                        f"about {subject}.\n\n{entry['question']}\n{opts}\nAnswer:"),
            "choices": [f" {l}" for l in _LETTERS],
            "answer": int(entry["answer"]),
        })
    return samples
