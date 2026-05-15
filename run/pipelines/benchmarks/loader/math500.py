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

def load_math500_dataset(query_version: str = "qwen"):    
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    raw = load_dataset("HuggingFaceH4/MATH-500")['test']
    examples = []
    for entry in raw:
        q_str = QUERY_TEMPLATE.format(Question=entry['problem'])
        a_str = entry['answer']
        sol_str = entry['solution']
        examples.append({
            "query": q_str,
            "solution": sol_str,
            "answer": a_str
        })
    return examples