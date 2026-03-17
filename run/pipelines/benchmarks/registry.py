"""Centralized benchmark registry with lazy loading.

Provides lazy-loaded benchmark dataset loaders to avoid importing
heavy dependencies (like `datasets`) until actually needed.
"""

from typing import Callable, List, Optional, Any
import random


AVAILABLE_BENCHMARKS = [    
    # math
    "gsm8k",
    "aime",
    "math-500",
    
    # code
    "human-eval",
    "mbpp",
    
    # longbench
    "hotpotqa",
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench_p",
    
    # others
    "mt-bench",
    "alpaca",
    "cnn-dm",
    "gpqa",
    "livecodebench",
]

_LOADER_CONFIG = {
    # TBD
    "mt-bench":             (".mtbench", "load_mtbench_dataset"),
    "alpaca":               (".alpaca", "load_alpaca_dataset"),
    "cnn-dm":               (".cnndm", "load_cnndm_dataset"),
    "gpqa":                 (".gpqa", "load_gpqa_dataset"),
    "livecodebench":        (".livecodebench", "load_livecodebench_dataset"),
    
    # math
    "gsm8k":                (".gsm8k", "load_gsm8k_dataset"),
    "aime":                 (".aime", "load_aime_dataset"),
    "math-500":             (".math500", "load_math500_dataset"),
    
    # code
    "human-eval":           (".humaneval", "load_humaneval_dataset"),
    "mbpp":                 (".mbpp", "load_mbpp_dataset"),
    
    # longbench
    "hotpotqa":             (".hotpotqa", "load_hotpotqa_dataset"),
    "narrativeqa":          (".narrativeqa", "load_narrativeqa_dataset"),
    "qasper":               (".qasper", "load_qasper_dataset"),
    "multifieldqa_en":      (".multifieldqa_en", "load_multifieldqa_en_dataset"),
    "2wikimqa":             ("._2wikimqa", "load_2wikimqa_dataset"),
    "musique":              (".musique", "load_musique_dataset"),
    "gov_report":           (".gov_report", "load_gov_report_dataset"),
    "qmsum":                (".qmsum", "load_qmsum_dataset"),
    "multi_news":           (".multi_news", "load_multi_news_dataset"),
    "trec":                 (".trec", "load_trec_dataset"),
    "triviaqa":             (".triviaqa", "load_triviaqa_dataset"),
    "samsum":               (".samsum", "load_samsum_dataset"),
    "passage_count":        (".passage_count", "load_passage_count_dataset"),
    "passage_retrieval_en": (".passage_retrieval_en", "load_passage_retrieval_en_dataset"),
    "lcc":                  (".lcc", "load_lcc_dataset"),
    "repobench_p":          (".repobench_p", "load_repobench_p_dataset"),
}


def _import_error_message(bench_name: str, package: str = "datasets") -> str:
    return (
        f"Benchmark '{bench_name}' requires the '{package}' library.\n"
        f"Install with: pip install {package}"
    )


def get_loader(bench_name: str, with_answers: bool = False) -> Callable:
    """Get a dataset loader function for the specified benchmark.
    
    Args:
        bench_name: Name of the benchmark (e.g., "mt-bench", "gsm8k")
        with_answers: If True, return loader that includes answer data (for accuracy eval)
        
    Returns:
        A callable that loads the dataset when invoked
        
    Raises:
        ValueError: If benchmark name is unknown or doesn't support with_answers
        ImportError: If required dependencies are not installed
    """
    if bench_name not in AVAILABLE_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: '{bench_name}'. "
            f"Available: {', '.join(AVAILABLE_BENCHMARKS)}"
        )
    
    module_name, dataset_loader = _LOADER_CONFIG[bench_name]
    
    try:
        import importlib
        module = importlib.import_module(module_name, package="run.pipelines.benchmarks.loader")
        return getattr(module, dataset_loader)
    except ImportError as e:
        raise ImportError(_import_error_message(bench_name)) from e


def extract_prompt(item: Any) -> str:
    """Extract prompt text from various benchmark formats.
    
    Handles:
        - str: Return as-is
        - list: MT-Bench format (list of turns), return first turn
        - dict: Try common keys (query, prompt, question, input, text)
    """
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return item[0] if item else ""
    if isinstance(item, dict):
        for key in ["query", "prompt", "question", "input", "text"]:
            if key in item:
                return item[key]
    return str(item)


def load_dataset(
    bench_name: str,
    max_samples: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True,
    query_version: str = "llama"
) -> List[Any]:
    """Load a benchmark dataset with optional shuffling and sampling.
    
    Args:
        bench_name: Name of the benchmark
        max_samples: Maximum number of samples to return (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle the dataset
        with_answers: If True, load dataset with answer data
        
    Returns:
        List of dataset items
    """
    loader = get_loader(bench_name)
    dataset = loader(query_version=query_version)
    
    if shuffle:
        random.seed(seed)
        random.shuffle(dataset)
    
    if max_samples is not None:
        dataset = dataset[:max_samples]
    
    return dataset


def validate_benchmarks(bench_list: List[str]) -> None:
    """Validate that all benchmark names are known and support requested mode.
    
    Args:
        bench_list: List of benchmark names to validate
        with_answers: If True, also validate that benchmarks support with_answers mode
        
    Raises:
        ValueError: If any benchmark name is unknown or doesn't support with_answers
    """
    unknown = [b for b in bench_list if b not in AVAILABLE_BENCHMARKS]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {', '.join(unknown)}. "
            f"Available: {', '.join(AVAILABLE_BENCHMARKS)}"
        )