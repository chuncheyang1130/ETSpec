# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QWEN_QUERY_TEMPLATE = r"""
Write a solution to the following problem and make sure that it passes the tests:
```python
{Prompt}
```
You should put your solution inside ```python ... ``` block.
""".strip()

LLAMA_QUERY_TEMPLATE = r"""
Write a solution to the following problem and make sure that it passes the tests:
```python
{Prompt}
```
You should put your solution inside ```python ... ``` block.
""".strip()

# HUMANEVAL
def load_humaneval_dataset(query_version: str = "llama"):
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    samples = []
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    for entry in dataset:
        samples.append({
            "query": QUERY_TEMPLATE.format(Prompt=entry['prompt']),
            "testcase": entry['test'],
            "entry_point": entry['entry_point'],
        })
    
    return samples