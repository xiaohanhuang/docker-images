"""
Evaluation component — evaluate a language model checkpoint on a test dataset.

Image: ml-gpu
"""

from typing import Dict, List, Optional

from flytekit import Resources, task
from flytekit.types.file import FlyteFile


@task(
    retries=1,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def evaluate_lm(
    checkpoint_path: FlyteFile,
    test_data: FlyteFile,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Evaluate a language model on a test dataset.

    Args:
        checkpoint_path: Path to the model checkpoint or LoRA adapter directory.
        test_data: Test dataset JSONL file.
        metrics: List of metric names to compute (e.g. ``["perplexity", "rouge"]``).

    Returns:
        Dictionary mapping metric names to their computed values.
    """
    import json
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if metrics is None:
        metrics = ["perplexity"]

    checkpoint_path.download()
    test_data.download()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path.path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path.path, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # Load test examples
    texts = []
    with open(test_data.path) as fh:
        for line in fh:
            item = json.loads(line.strip())
            texts.append(item.get("text", ""))

    results: Dict[str, float] = {}

    if "perplexity" in metrics:
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in texts[:100]:  # cap at 100 for speed
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        results["perplexity"] = math.exp(total_loss / total_tokens) if total_tokens else 0.0

    if "rouge" in metrics:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1_scores, rl_scores = [], []
        with torch.no_grad():
            for text in texts[:50]:
                half = len(text) // 2
                inputs = tokenizer(text[:half], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=50)
                generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                scores = scorer.score(text, generated)
                r1_scores.append(scores["rouge1"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)
        results["rouge1"] = sum(r1_scores) / len(r1_scores) if r1_scores else 0.0
        results["rougeL"] = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0

    return results
