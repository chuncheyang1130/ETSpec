# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from random import sample

from datasets import load_dataset

QWEN_QUERY_TEMPLATE = r"""
Question: {Question}
Please reason step by step, and put your final answer within \boxed{{}}.
""".strip()

LLAMA_QUERY_TEMPLATE = r"""
Given the following problem, reason and give a final answer to the problem.
Problem: {Question}
Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
""".strip()

# GSM8K
def load_gsm8k_dataset(query_version: str = "llama"):
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    samples = []
    dataset = load_dataset("openai/gsm8k", "main")
    
    for entry in dataset['test']:
        samples.append({
            "query": QUERY_TEMPLATE.format(Question=entry['question']),
            "answer": entry['answer']
        })
    
    return samples