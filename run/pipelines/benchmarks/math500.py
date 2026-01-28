# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

REASONING_QUERY_TEMPLATE = r"""
Given the following problem, reason and give a final answer to the problem.
Question: {Question}
Please reason step by step, and put your final answer within \boxed{{}}.
""".strip()

GENERAL_QUERY_TEMPLATE = r"""
Given the following problem, reason and give a final answer to the problem.
Problem: {Question}
Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
""".strip()

# MATH-500
def load_math500_dataset():
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    formatted_dataset = [REASONING_QUERY_TEMPLATE.format(Question=entry['problem']) for entry in dataset['test']]
    return formatted_dataset

def load_math500_dataset_answer(query_version: str = "reasoning"):
    raw = load_dataset("HuggingFaceH4/MATH-500")['test']
    examples = []
    for entry in raw:
        if query_version == "reasoning":
            q_str = REASONING_QUERY_TEMPLATE.format(Question=entry['problem'])
        else:
            q_str = GENERAL_QUERY_TEMPLATE.format(Question=entry['problem'])
        a_str = entry['answer']
        sol_str = entry['solution']
        examples.append({
            "question": q_str,
            "solution": sol_str,
            "answer": a_str
        })
    return examples