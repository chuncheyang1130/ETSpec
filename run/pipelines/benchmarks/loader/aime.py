# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QWEN_QUERY_TEMPLATE = r"""
Given the following problem, reason and give a final answer to the problem.
Question: {Question}
Please reason step by step, and put your final answer within \boxed{{}}.
""".strip()

LLAMA_QUERY_TEMPLATE = r"""
Given the following problem, reason and give a final answer to the problem.
Problem: {Question}
Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
""".strip()

def load_aime_dataset(query_version: str = "qwen"):
    """
    Returns list of dicts with 'query' and 'solution' for AIME‑2024.
    """
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    raw = load_dataset("HuggingFaceH4/aime_2024", split="train")
    examples = []
    for entry in raw:
        q_str = QUERY_TEMPLATE.format(Question=entry["problem"])
        a_str = entry["answer"]  
        sol_str = entry["solution"]
        examples.append({
            "query": q_str, 
            "answer": a_str,
            "solution": sol_str
        })
    return examples
