# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QWEN_QUERY_TEMPLATE = r"""
{Prompt}
Your code should satisfy the following assertion:
{Example}
You should put your solution inside ```python ... ``` block.
""".strip()

LLAMA_QUERY_TEMPLATE = r"""
{Prompt}
Your code should satisfy the following assertion:
{Example}
You should put your solution inside ```python ... ``` block.
""".strip()

# MBPP
def load_mbpp_dataset(query_version: str = "llama"):
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    samples = []
    dataset = load_dataset("evalplus/mbppplus", split="test")
    
    for entry in dataset:
        example_code = "\n".join(entry['test_list'])
        samples.append({
            "query": QUERY_TEMPLATE.format(Prompt=entry['prompt'], Example=example_code),
            "testcase": entry['test']
        })
    
    return samples