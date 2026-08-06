import random
from datasets import load_dataset

# Keep the same fluent QUERY_TEMPLATE as before
QUERY_TEMPLATE = r"""
Provide your step-by-step reasoning. On the last line by itself, give the final answer in the format "Answer: <LETTER>".
Question: {Question}
Options:
{Options}

Answer:
""".strip()

def load_mmlu_pro_dataset(query_version: str = "llama"):
    samples = []
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    for entry in dataset:
        options_str = "\n".join(
            f"({chr(ord('A') + i)}) {opt}"
            for i, opt in enumerate(entry['options'])
        )
        
        answer_letter = chr(ord('A') + entry['answer_index'])
        samples.append({
            "query": QUERY_TEMPLATE.format(Question=entry['question'], Options=options_str), 
            "answer": answer_letter
        })
    
    return samples