# Prompt template adapted from https://github.com/openai/simple-evals/tree/main
from datasets import load_dataset

QWEN_QUERY_TEMPLATE = r"""

""".strip()

LLAMA_QUERY_TEMPLATE = r"""
Write a solution to the following problem and make sure that it passes the tests:
```python
{Prompt}
```
Here is the completed function:
""".strip()

# HUMANEVAL
def load_humaneval_dataset(query_version: str = "llama"):
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    dataset = load_dataset("openai/openai_humaneval", split="test")
    formatted_dataset = [QUERY_TEMPLATE.format(Prompt=entry['prompt']) for entry in dataset]
    
    return formatted_dataset

def load_humaneval_dataset_answer(query_version: str = "llama"):
    if query_version == "qwen":
        QUERY_TEMPLATE = QWEN_QUERY_TEMPLATE
    elif query_version == "llama":
        QUERY_TEMPLATE = LLAMA_QUERY_TEMPLATE
    else:
        raise ValueError(f"Unknown query_version: {query_version}")
    
    examples = []
    dataset = load_dataset("openai/openai_humaneval", split="test")
    for entry in dataset:
        prompt = QUERY_TEMPLATE.format(Prompt=entry['prompt'])
        solution = entry['canonical_solution']  # the canonical solution string :contentReference[oaicite:0]{index=0}
        testcase = entry['test']
        examples.append({"prompt": prompt, "solution": solution, "testcase": testcase})
        
    return examples