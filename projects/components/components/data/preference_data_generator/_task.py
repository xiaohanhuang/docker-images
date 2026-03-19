"""Flyte task definition for preference_data_generator component."""

from datetime import timedelta
from typing import Optional, Tuple

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from ._generators import _get_generator
from ._judges import _get_judge


@task(
    retries=1,
    timeout=timedelta(hours=4),
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(cpu="16", mem="64Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def generate_preference_data(
    prompt_data_path: FlyteFile,
    generator_model: str,
    prompt_column: str = "prompt",
    judge_model: str = "gpt-4o",
    n_candidates: int = 4,
    judge_criteria: str = "helpfulness, accuracy, safety",
    output_format: str = "dpo",
    scored_data_path: Optional[str] = None,
) -> Tuple[FlyteFile, int, float]:
    """Generate preference pairs (chosen/rejected) for DPO/RLHF training using AI judge.

    Supports two modes:

    1. **Generation mode** (default): Generates N responses from the target model for each
       prompt, then uses a judge LLM to rank them into preference pairs.
    2. **Conversion mode** (``scored_data_path`` provided): Converts existing pre-scored
       candidate data into DPO/ranking format, skipping generation and judging entirely.

    The output preference dataset is returned as a FlyteFile so Flyte handles
    S3 upload to its managed data bucket automatically.

    Args:
        prompt_data_path: S3 URI to prompt dataset (JSONL format with prompt column).
            Used only in generation mode (ignored when ``scored_data_path`` is set).
        generator_model: HuggingFace model ID or vLLM endpoint to generate candidate responses.
        prompt_column: Column name containing prompts in the input JSONL.
        judge_model: Judge model for ranking responses. Supports:
            - OpenAI models (e.g., "gpt-4o", "gpt-4-turbo")
            - vLLM endpoints (format: "vllm://model-name@http://endpoint:port")
            - HuggingFace models loaded locally (format: "hf://model-id")
            - "heuristic" for a simple length/quality heuristic judge
        n_candidates: Number of responses to generate per prompt.
        judge_criteria: Comma-separated criteria for ranking (e.g., "helpfulness, accuracy").
        output_format: Output format - "dpo" for chosen/rejected pairs or "ranking" for
            full ranking with scores.
        scored_data_path: Optional S3 URI to pre-scored candidate data (conversion mode).
            When provided, the component reads this file instead of generating responses.
            Expected JSONL format (one JSON object per line)::

                {"prompt": "...", "responses": [{"text": "...", "score": 8.5}, ...]}

    Returns:
        Tuple of (preference_file, num_pairs, avg_score_delta):
            - preference_file: FlyteFile pointing to the preference dataset (JSONL)
            - num_pairs: Number of preference pairs created
            - avg_score_delta: Average score difference between chosen and rejected
              responses (proxy for preference quality)
    """
    import json
    import os

    import s3fs

    # Convert FlyteFile input to S3 path string
    prompt_data_path = getattr(prompt_data_path, "remote_source", None) or str(prompt_data_path)  # type: ignore[assignment]

    # Validate inputs
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be at least 2, got {n_candidates}")
    if output_format not in ["dpo", "ranking"]:
        raise ValueError(f"Invalid output_format '{output_format}'. Must be 'dpo' or 'ranking'")

    s3 = s3fs.S3FileSystem()

    # Download input prompts
    os.makedirs("/tmp/inputs", exist_ok=True)
    input_file = "/tmp/inputs/prompts.jsonl"
    s3.get(prompt_data_path, input_file)

    # Load prompts
    prompts = []
    with open(input_file) as fh:
        for line_num, line in enumerate(fh, 1):
            try:
                item = json.loads(line.strip())
                if prompt_column not in item:
                    raise ValueError(f"Line {line_num}: Missing required column '{prompt_column}'")
                prompts.append(item[prompt_column])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num}: Invalid JSON - {exc}") from exc

    if not prompts:
        raise ValueError(f"No prompts found in {prompt_data_path}")

    # Build the list of (prompt, ranked_candidates) — generation or conversion mode
    all_ranked: list[tuple[str, list[dict]]] = []

    if scored_data_path:
        # ── Conversion mode: read pre-scored candidates from S3 ────────────────
        scored_file = "/tmp/inputs/scored.jsonl"
        s3.get(scored_data_path, scored_file)

        with open(scored_file) as fh:
            for line_num, line in enumerate(fh, 1):
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError as exc:
                    raise ValueError(f"scored_data: line {line_num}: invalid JSON – {exc}") from exc
                for field in ("prompt", "responses"):
                    if field not in item:
                        raise ValueError(
                            f"scored_data: line {line_num}: missing required field '{field}'"
                        )
                ranked = sorted(
                    [{"response": r["text"], "score": r["score"]} for r in item["responses"]],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                all_ranked.append((item["prompt"], ranked))
    else:
        # ── Generation mode: generate candidates then rank with judge ──────────
        generator = _get_generator(generator_model)
        judge = _get_judge(judge_model)

        print(f"Generating {n_candidates} candidates for {len(prompts)} prompts...")
        for i, prompt in enumerate(prompts):
            candidates = [generator(prompt) for _ in range(n_candidates)]
            ranked_candidates = judge(prompt, candidates, judge_criteria)
            all_ranked.append((prompt, ranked_candidates))
            if (i + 1) % 50 == 0 or i + 1 == len(prompts):
                print(f"  Progress: {i + 1}/{len(prompts)} prompts processed")

    # Write preference pairs / rankings to output file
    output_file = "/tmp/preference_data.jsonl"
    num_pairs = 0
    total_score_delta = 0.0

    with open(output_file, "w") as out_f:
        for prompt, ranked_candidates in all_ranked:
            if output_format == "dpo":
                # For DPO: create pairs of (best, worst) from ranked candidates
                best = ranked_candidates[0]
                worst = ranked_candidates[-1]

                preference_pair = {
                    "prompt": prompt,
                    "chosen": best["response"],
                    "rejected": worst["response"],
                    "chosen_score": best["score"],
                    "rejected_score": worst["score"],
                }
                out_f.write(json.dumps(preference_pair) + "\n")
                num_pairs += 1
                total_score_delta += best["score"] - worst["score"]

            else:  # output_format == "ranking"
                # For ranking: output full ranking with all scores
                ranking_entry = {
                    "prompt": prompt,
                    "candidates": [
                        {"response": c["response"], "score": c["score"], "rank": idx + 1}
                        for idx, c in enumerate(ranked_candidates)
                    ],
                }
                out_f.write(json.dumps(ranking_entry) + "\n")
                num_pairs += 1
                # Use top vs bottom for score delta
                total_score_delta += ranked_candidates[0]["score"] - ranked_candidates[-1]["score"]

    # Return FlyteFile — Flyte automatically uploads the local file to its
    # managed S3 bucket, avoiding IAM permission issues.
    avg_score_delta = total_score_delta / num_pairs if num_pairs > 0 else 0.0

    return FlyteFile(path=output_file), num_pairs, avg_score_delta
