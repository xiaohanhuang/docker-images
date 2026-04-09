"""Flyte task definition for reward_bench component."""

from typing import Dict, List, Optional

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory

# ── Category → subset mapping (single source of truth) ───────────────
CATEGORY_MAP: Dict[str, List[str]] = {
    "chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-medium",
    ],
    "chat-hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


@task(
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def benchmark_reward_model(
    reward_model_path: FlyteDirectory,
    benchmark: str = "rewardbench",
    categories: Optional[List[str]] = None,
    batch_size: int = 8,
) -> dict:
    """Benchmark a reward model against RewardBench evaluation suite.

    Evaluates how well a reward model discriminates between good and bad outputs
    across standardized preference datasets. Returns accuracy scores per category
    and overall, plus comparison against published baselines.

    Args:
        reward_model_path: S3 path or HuggingFace model ID for the reward model.
        benchmark: Benchmark suite name (default: "rewardbench").
        categories: Specific categories to evaluate (None = all). Valid categories:
            "chat", "chat-hard", "safety", "reasoning" (includes code+math).
        batch_size: Inference batch size for evaluation.

    Returns:
        Dictionary containing:
            - overall_accuracy (float): Overall preference accuracy across all categories
            - category_scores (dict): Accuracy per category (chat, safety, reasoning, etc.)
            - comparison (dict): Comparison against published baseline models
            - mlflow_run_id (str): MLflow run ID for tracking

    Example:
        >>> result = benchmark_reward_model(
        ...     reward_model_path="OpenAssistant/reward-model-deberta-v3-large-v2",
        ...     categories=["chat", "safety"],
        ...     batch_size=16
        ... )
        >>> print(f"Overall accuracy: {result['overall_accuracy']:.2%}")
    """
    import tempfile

    # Convert FlyteDirectory input to S3 path string
    reward_model_path = getattr(reward_model_path, "remote_source", None) or str(reward_model_path)  # type: ignore[assignment]

    import mlflow
    import torch
    from datasets import load_dataset

    from ._model_loading import _find_local_model_root, _load_tokenizer_and_model

    # ── Load reward model ─────────────────────────────────────────────────
    # Support both HF model IDs and S3 paths
    if reward_model_path.startswith("s3://"):
        import s3fs

        s3 = s3fs.S3FileSystem()

        # Download model to temp directory
        model_dir = tempfile.mkdtemp()
        # Strip trailing slash from the S3 path
        s3_path = reward_model_path.rstrip("/")
        s3.get(s3_path, model_dir, recursive=True)

        # s3fs.get with recursive=True may nest files under a subdirectory.
        # Also, LoRA adapter checkpoints may not contain config.json, so look
        # for either config.json, adapter_config.json, or tokenizer files.
        model_path = _find_local_model_root(model_dir)
    else:
        model_path = reward_model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _load_tokenizer_and_model(model_path, device)
    model.eval()

    # ── Load RewardBench dataset ──────────────────────────────────────────
    if benchmark == "rewardbench":
        dataset_name = "allenai/reward-bench"
    elif benchmark == "rewardbench-2":
        dataset_name = "allenai/reward-bench-2"
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Supported: 'rewardbench', 'rewardbench-2'"
        )

    # RewardBench v1 uses 'filtered' split (not 'test')
    try:
        dataset = load_dataset(dataset_name, split="filtered")
    except ValueError:
        # Fallback: try 'test' split for forward compatibility
        dataset = load_dataset(dataset_name, split="test")

    # Filter by categories if specified
    if categories is not None:
        subsets_to_keep = []
        for cat in categories:
            if cat not in CATEGORY_MAP:
                raise ValueError(f"Unknown category: {cat}. Valid: {list(CATEGORY_MAP.keys())}")
            subsets_to_keep.extend(CATEGORY_MAP[cat])

        dataset = dataset.filter(lambda x: x.get("subset") in subsets_to_keep)

    # ── Evaluate reward model ─────────────────────────────────────────────
    category_results: Dict[str, Dict[str, int]] = {}
    all_correct: list[bool] = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        prompts = batch["prompt"]
        chosen_texts = batch["chosen"]
        rejected_texts = batch["rejected"]
        subsets = batch.get("subset", ["unknown"] * len(prompts))

        # Score chosen and rejected completions in a single batched forward pass
        chosen_texts_full = [p + c for p, c in zip(prompts, chosen_texts)]
        rejected_texts_full = [p + r for p, r in zip(prompts, rejected_texts)]
        all_texts = chosen_texts_full + rejected_texts_full

        combined_inputs = tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        combined_inputs = {k: v.to(device) for k, v in combined_inputs.items()}

        with torch.no_grad():
            logits = model(**combined_inputs).logits
            # Enforce scalar reward output (single logit per example)
            if logits.dim() > 1 and logits.size(-1) != 1:
                raise RuntimeError(
                    f"Reward model outputs {logits.size(-1)} logits per example; "
                    "expected a single scalar reward (num_labels=1). "
                    "Multi-class reward heads are not supported."
                )
            all_scores = logits.squeeze(-1).cpu().tolist()

        # Split scores back into chosen and rejected
        num_examples = len(prompts)
        chosen_scores = all_scores[:num_examples]
        rejected_scores = all_scores[num_examples:]

        # Normalize to lists (handles single-element batches)
        if isinstance(chosen_scores, float):
            chosen_scores = [chosen_scores]
        if isinstance(rejected_scores, float):
            rejected_scores = [rejected_scores]

        for c_score, r_score, subset in zip(chosen_scores, rejected_scores, subsets):
            is_correct = c_score > r_score
            all_correct.append(is_correct)

            if subset not in category_results:
                category_results[subset] = {"correct": 0, "total": 0}
            category_results[subset]["total"] += 1
            if is_correct:
                category_results[subset]["correct"] += 1

    # ── Compute metrics ───────────────────────────────────────────────────
    overall_accuracy = sum(all_correct) / len(all_correct) if all_correct else 0.0

    category_scores = {
        subset: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for subset, data in category_results.items()
    }

    # Aggregate by high-level category (reuses module-level CATEGORY_MAP)
    high_level_scores: Dict[str, float] = {}
    for high_level, cat_subsets in CATEGORY_MAP.items():
        subset_scores = [category_scores.get(s, 0.0) for s in cat_subsets if s in category_scores]
        if subset_scores:
            high_level_scores[high_level] = sum(subset_scores) / len(subset_scores)

    # ── Comparison with baselines ─────────────────────────────────────────
    # Published baseline scores from RewardBench leaderboard (approximate)
    baselines = {
        "Starling-RM-34B": 0.81,
        "Tulu-2-DPO-70B": 0.79,
        "OpenAssistant-RM-Deberta-v3-Large": 0.73,
        "GPT-4-0613": 0.86,
        "Claude-2": 0.84,
    }

    comparison = {
        "model_score": overall_accuracy,
        "baselines": baselines,
        "rank_among_baselines": sum(1 for v in baselines.values() if v < overall_accuracy),
    }

    # ── Log to MLflow ─────────────────────────────────────────────────────
    mlflow.set_experiment("reward-model-benchmarking")
    with mlflow.start_run():
        mlflow.log_param("model_path", reward_model_path)
        mlflow.log_param("benchmark", benchmark)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("categories", categories or "all")

        mlflow.log_metric("overall_accuracy", overall_accuracy)
        for cat, score in high_level_scores.items():
            mlflow.log_metric(f"accuracy_{cat}", score)

        for subset, score in category_scores.items():
            mlflow.log_metric(f"subset_{subset}", score)

        run_id = mlflow.active_run().info.run_id

    return {
        "overall_accuracy": overall_accuracy,
        "category_scores": high_level_scores,
        "detailed_scores": category_scores,
        "comparison": comparison,
        "mlflow_run_id": run_id,
    }
