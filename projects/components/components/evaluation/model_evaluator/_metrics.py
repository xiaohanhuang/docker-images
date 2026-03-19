"""Model evaluation metric implementations (perplexity, ROUGE, accuracy)."""

from typing import Any, Dict, List


def compute_metrics(
    model: Any,
    tokenizer: Any,
    test_dataset: Any,
    metrics: List[str],
    device: str,
) -> Dict[str, float]:
    """Compute requested evaluation metrics on the test dataset.

    Args:
        model: Loaded language model (eval mode).
        tokenizer: Corresponding tokenizer.
        test_dataset: HuggingFace dataset with input_ids and attention_mask.
        metrics: List of metrics to compute (perplexity, rouge, accuracy).
        device: Device string ('cuda' or 'cpu').

    Returns:
        Dictionary mapping metric names to values.
    """
    import math

    import torch

    results: Dict[str, float] = {}

    # Perplexity
    if "perplexity" in metrics:
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for i, example in enumerate(test_dataset):
                if i >= 100:
                    break
                inputs = {
                    "input_ids": torch.tensor([example["input_ids"]]).to(device),
                    "attention_mask": torch.tensor([example["attention_mask"]]).to(device),
                }
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        results["perplexity"] = (
            math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        )

    # ROUGE
    if "rouge" in metrics:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1_scores, rl_scores = [], []

        with torch.no_grad():
            for i, example in enumerate(test_dataset):
                if i >= 50:
                    break

                input_ids = torch.tensor([example["input_ids"][:256]]).to(device)
                attention_mask = torch.tensor([example["attention_mask"][:256]]).to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                reference = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

                scores = scorer.score(reference, generated)
                r1_scores.append(scores["rouge1"].fmeasure)
                rl_scores.append(scores["rougeL"].fmeasure)

        results["rouge1"] = sum(r1_scores) / len(r1_scores) if r1_scores else 0.0
        results["rougeL"] = sum(rl_scores) / len(rl_scores) if rl_scores else 0.0

    # Accuracy (exact match)
    if "accuracy" in metrics:
        correct = 0
        total = 0
        with torch.no_grad():
            for i, example in enumerate(test_dataset):
                if i >= 100:
                    break

                input_ids = torch.tensor([example["input_ids"][:256]]).to(device)
                attention_mask = torch.tensor([example["attention_mask"][:256]]).to(device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                )
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                reference = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

                if generated.strip() == reference.strip():
                    correct += 1
                total += 1

        results["accuracy"] = correct / total if total > 0 else 0.0

    return results
