# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
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
    dataset = load_dataset("openai/gsm8k", "main")
    if query_version == "qwen":
        formatted_dataset = [QWEN_QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    elif query_version == "llama":
        formatted_dataset = [LLAMA_QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    else:
        raise ValueError(f"Unknown query_version: {query_version}") 
    return formatted_dataset

def load_gsm8k_dataset_answer(query_version: str = "llama"):
    raw = load_dataset("openai/gsm8k", "main")['test']
    examples = []
    for entry in raw:
        if query_version == "qwen":
            q_str = QWEN_QUERY_TEMPLATE.format(Question=entry['question'])
        elif query_version == "llama":
            q_str = LLAMA_QUERY_TEMPLATE.format(Question=entry['question'])
        else:
            raise ValueError(f"Unknown query_version: {query_version}")
        a_str = entry['answer']
        examples.append({
            "question": q_str,
            "answer": a_str
        })
    return examples