# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

REASONING_QUERY_TEMPLATE= """
Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{Question}
""".strip()

GENERAL_QUERY_TEMPLATE = """
Given the following problem, reason and give a final answer to the problem.\nProblem: {Question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n
""".strip()

# GSM8K
def load_gsm8k_dataset(query_version: str = "reasoning"):
    dataset = load_dataset("openai/gsm8k", "main")
    if query_version == "reasoning":
        formatted_dataset = [REASONING_QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    else:
        formatted_dataset = [GENERAL_QUERY_TEMPLATE.format(Question=entry['question']) for entry in dataset['test']]
    return formatted_dataset

def load_gsm8k_dataset_answer(query_version: str = "reasoning"):
    raw = load_dataset("openai/gsm8k", "main")['test']
    examples = []
    for entry in raw:
        if query_version == "reasoning":
            q_str = REASONING_QUERY_TEMPLATE.format(Question=entry['question'])
        else:
            q_str = GENERAL_QUERY_TEMPLATE.format(Question=entry['question'])
        a_str = entry['answer']
        examples.append({
            "question": q_str,
            "answer": a_str
        })
    return examples