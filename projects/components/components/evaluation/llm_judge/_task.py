"""Flyte task definition for llm_judge component."""

import json
from typing import Dict, List, Optional

from flytekit import Resources, task
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile

from ._backends import (
    _build_evaluation_prompt,
    _call_anthropic_judge,
    _call_openai_judge,
    _call_vllm_judge,
)
from ._evaluation import _heuristic_evaluate, _load_predictions_for_llm_judge
from ._rubrics import DEFAULT_RUBRICS


@task(
    retries=2,
    requests=Resources(cpu="2", mem="8Gi"),
    limits=Resources(cpu="4", mem="16Gi"),
    cache=False,
)
def llm_judge(
    predictions_path: FlyteDirectory,
    ground_truth_path: Optional[FlyteFile] = None,
    judge_model: str = "gpt-4o",
    scorers: Optional[List[str]] = None,
    custom_rubric: Optional[str] = None,
    sample_size: int = 100,
    thresholds: Optional[Dict[str, float]] = None,
    mlflow_experiment: Optional[str] = None,
) -> dict:
    """Evaluate model predictions using LLM-as-Judge.

    Args:
        predictions_path: S3 path to JSONL file with model predictions,
            or S3 path to a model checkpoint directory.
        ground_truth_path: Optional S3 path to JSONL file with ground truth.
            Format: {{"input": "...", "ground_truth": "...", "id": "..."}}
        judge_model: Judge model identifier. Supports:
            - "heuristic": Simple text-quality heuristics (no LLM needed).
            - OpenAI: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
            - Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
                        - vLLM endpoint: "http://localhost:8000/v1" (OpenAI-compatible)
                        - vLLM endpoint with explicit model:
                            "vllm://model-name@http://localhost:8000/v1"
        scorers: List of scoring dimensions. Options: "relevance", "coherence",
            "hallucination", "toxicity", "helpfulness", "safety", "accuracy",
            "custom". Default: ["relevance", "coherence"]
        custom_rubric: Custom evaluation rubric prompt (only used if "custom" in scorers)
        sample_size: Number of examples to evaluate (random sample). Default: 100
        thresholds: Optional dict of minimum passing scores per dimension
            (e.g., {{"relevance": 3.5}})
        mlflow_experiment: Optional MLflow experiment name for logging results

    Returns:
        Dictionary with aggregated metrics:
            - <scorer>_mean: Average score for each scorer
            - <scorer>_pass_rate: Fraction of examples above threshold
            - overall_pass_rate: Fraction of examples passing all thresholds
            - num_evaluated: Number of examples evaluated
            - detailed_results_path: S3 path to per-example scores
    """
    import os
    import tempfile

    # Convert Flyte types to S3 path strings
    predictions_path = getattr(predictions_path, "remote_source", None) or str(predictions_path)  # type: ignore[assignment]
    if ground_truth_path is not None:
        ground_truth_path = (  # type: ignore[assignment]
            getattr(ground_truth_path, "remote_source", None) or str(ground_truth_path)
        )

    if scorers is None:
        scorers = ["relevance", "coherence"]

    if thresholds is None:
        thresholds = {scorer: 3.0 for scorer in scorers}

    # ── Validate scorers and rubrics early (before S3 access) ─────────────
    rubrics = {}
    for scorer in scorers:
        if scorer == "custom":
            if not custom_rubric:
                raise ValueError("custom_rubric must be provided when 'custom' is in scorers")
            rubrics["custom"] = custom_rubric
        else:
            if scorer not in DEFAULT_RUBRICS:
                valid_options = list(DEFAULT_RUBRICS.keys())
                raise ValueError(f"Unknown scorer: {scorer}. Valid options: {valid_options}")
            rubrics[scorer] = DEFAULT_RUBRICS[scorer]

    import s3fs

    s3 = s3fs.S3FileSystem()

    # ── Heuristic mode: skip LLM calls, return text-quality metrics ───────
    if judge_model == "heuristic":
        return _heuristic_evaluate(
            predictions_path=predictions_path,
            scorers=scorers,
            thresholds=thresholds,
            sample_size=sample_size,
            s3=s3,
            mlflow_experiment=mlflow_experiment,
        )

    # Load predictions (JSONL file or checkpoint directory)
    predictions = _load_predictions_for_llm_judge(
        predictions_path=predictions_path,
        sample_size=sample_size,
        s3=s3,
    )

    # Load ground truth if provided
    ground_truth_map = {}
    if ground_truth_path:
        with s3.open(ground_truth_path, "r") as fh:
            gt_lines = fh.read().split("\n")
        ground_truths = [json.loads(line) for line in gt_lines if line.strip()]
        ground_truth_map = {item["id"]: item.get("ground_truth", "") for item in ground_truths}

    print(f"📊 Evaluating {len(predictions)} examples with judge model: {judge_model}")

    # Determine judge backend and normalize model/target values
    judge_api = "openai"
    judge_target = judge_model
    judge_backend_model = "default"

    if judge_model.startswith("vllm://"):
        spec = judge_model.removeprefix("vllm://")
        if "@" not in spec:
            raise ValueError(
                f"Invalid vLLM spec '{judge_model}'. Expected format: "
                "'vllm://model-name@http://endpoint:port'"
            )
        judge_backend_model, judge_target = spec.split("@", 1)
        judge_api = "vllm"
    elif judge_model.startswith("http://") or judge_model.startswith("https://"):
        judge_api = "vllm"
        judge_target = judge_model
    elif judge_model.startswith("claude-"):
        judge_api = "anthropic"

    # Evaluate each example
    detailed_results = []
    for example in predictions:
        example_id = example.get("id", "")
        input_text = example.get("input", "")
        prediction = example.get("prediction", "")
        ground_truth = ground_truth_map.get(example_id, None)

        result = {
            "id": example_id,
            "input": input_text,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "scores": {},
        }

        # Score on each dimension
        for scorer in scorers:
            rubric = rubrics[scorer]
            prompt = _build_evaluation_prompt(scorer, prediction, input_text, ground_truth, rubric)

            try:
                if judge_api == "openai":
                    api_key = os.environ.get("OPENAI_API_KEY")
                    response = _call_openai_judge(judge_target, prompt, api_key)
                elif judge_api == "anthropic":
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    response = _call_anthropic_judge(judge_target, prompt, api_key)
                else:  # vllm
                    response = _call_vllm_judge(
                        judge_target,
                        prompt,
                        model=judge_backend_model,
                    )

                score = float(response.get("score", 0))
                reasoning = response.get("reasoning", "")
                result["scores"][scorer] = {"score": score, "reasoning": reasoning}
            except Exception as e:
                print(f"⚠️  Error evaluating example {example_id} on {scorer}: {e}")
                result["scores"][scorer] = {"score": 0.0, "reasoning": f"Error: {str(e)}"}

        detailed_results.append(result)

    # Compute aggregate metrics
    metrics = {"num_evaluated": len(detailed_results)}

    for scorer in scorers:
        scores = [r["scores"][scorer]["score"] for r in detailed_results]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        metrics[f"{scorer}_mean"] = mean_score

        # Pass rate: fraction of examples above threshold
        threshold = thresholds.get(scorer, 3.0)
        pass_count = sum(1 for s in scores if s >= threshold)
        metrics[f"{scorer}_pass_rate"] = pass_count / len(scores) if scores else 0.0

    # Overall pass rate: fraction passing ALL thresholds
    overall_pass_count = 0
    for result in detailed_results:
        threshold_checks = [
            result["scores"][scorer]["score"] >= thresholds.get(scorer, 3.0) for scorer in scorers
        ]
        passes_all = all(threshold_checks)
        if passes_all:
            overall_pass_count += 1
    if detailed_results:
        metrics["overall_pass_rate"] = overall_pass_count / len(detailed_results)
    else:
        metrics["overall_pass_rate"] = 0.0

    # Save detailed results locally; Flyte will handle S3 upload if needed.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for result in detailed_results:
            tmp.write(json.dumps(result) + "\n")
        tmp_path = tmp.name

    metrics["detailed_results_path"] = tmp_path

    # Log to MLflow if requested
    if mlflow_experiment:
        import mlflow

        mlflow_tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow.monitoring.svc.cluster.local"
        )
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "judge_model": judge_model,
                    "scorers": ",".join(scorers),
                    "sample_size": sample_size,
                }
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            # Log sample results table
            import pandas as pd

            sample_results = []
            for result in detailed_results[:20]:
                row = {
                    "id": result["id"],
                    "input": result["input"][:100],
                    "prediction": result["prediction"][:100],
                }
                for scorer in scorers:
                    row[f"{scorer}_score"] = result["scores"][scorer]["score"]
                sample_results.append(row)

            df = pd.DataFrame(sample_results)
            mlflow.log_table(data=df, artifact_file="judge_sample_results.json")

            if tmp_path:
                mlflow.log_param("detailed_results_path", tmp_path)

    print("✅ Evaluation complete. Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    return metrics
