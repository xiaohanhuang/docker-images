"""Judge backends for ranking candidate responses in preference data generation."""


def _get_judge(model_spec: str):
    """Create a judge function for ranking responses."""

    def judge_with_openai(model_name: str):
        """Judge using OpenAI API."""
        import os

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI judge")

        client = OpenAI(api_key=api_key)

        def judge(prompt: str, candidates: list[str], criteria: str) -> list[dict]:
            # Create a judge prompt to rank candidates
            judge_prompt = f"""You are an expert judge evaluating AI-generated responses.

Prompt: {prompt}

Evaluate the following {len(candidates)} candidate responses based on these criteria: {criteria}

"""
            for i, candidate in enumerate(candidates, 1):
                judge_prompt += f"\nCandidate {i}:\n{candidate}\n"

            judge_prompt += f"""
Please rank these candidates from best to worst based on {criteria}.
For each candidate, provide a score from 0-10 and a brief justification.

Respond in JSON format:
{{
  "rankings": [
    {{"candidate": 1, "score": 8.5, "reason": "..."}},
    {{"candidate": 2, "score": 7.0, "reason": "..."}},
    ...
  ]
}}
"""

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI judge."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.2,  # Lower temperature for more consistent judgments
            )

            # Parse judge response
            import json

            judge_response = response.choices[0].message.content
            # Extract JSON from response (it might have markdown code blocks)
            if "```json" in judge_response:
                json_str = judge_response.split("```json")[1].split("```")[0].strip()
            elif "```" in judge_response:
                json_str = judge_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = judge_response.strip()

            rankings_data = json.loads(json_str)
            rankings = rankings_data["rankings"]

            # Sort by score (descending)
            rankings.sort(key=lambda x: x["score"], reverse=True)

            # Map back to candidates with scores; validate judge response fields and indices
            ranked_candidates = []
            for rank in rankings:
                if "candidate" not in rank or "score" not in rank:
                    raise ValueError(
                        f"Judge response missing required fields ('candidate', 'score'). "
                        f"Got: {rank}"
                    )
                candidate_idx = rank["candidate"] - 1  # Convert to 0-indexed
                if candidate_idx < 0 or candidate_idx >= len(candidates):
                    raise ValueError(
                        f"Judge returned invalid candidate index {rank['candidate']} "
                        f"(expected 1\u2013{len(candidates)})"
                    )
                ranked_candidates.append(
                    {
                        "response": candidates[candidate_idx],
                        "score": rank["score"],
                    }
                )

            return ranked_candidates

        return judge

    def judge_with_vllm(endpoint_url: str, model_name: str):
        """Judge using vLLM endpoint (same logic as OpenAI)."""
        from openai import OpenAI

        client = OpenAI(base_url=endpoint_url, api_key="not-required")

        def judge(prompt: str, candidates: list[str], criteria: str) -> list[dict]:
            # Same judge prompt as OpenAI
            judge_prompt = f"""You are an expert judge evaluating AI-generated responses.

Prompt: {prompt}

Evaluate the following {len(candidates)} candidate responses based on these criteria: {criteria}

"""
            for i, candidate in enumerate(candidates, 1):
                judge_prompt += f"\nCandidate {i}:\n{candidate}\n"

            judge_prompt += f"""
Please rank these candidates from best to worst based on {criteria}.
For each candidate, provide a score from 0-10 and a brief justification.

Respond in JSON format:
{{
  "rankings": [
    {{"candidate": 1, "score": 8.5, "reason": "..."}},
    {{"candidate": 2, "score": 7.0, "reason": "..."}},
    ...
  ]
}}
"""

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI judge."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.2,
            )

            # Parse judge response
            import json

            judge_response = response.choices[0].message.content
            # Extract JSON from response
            if "```json" in judge_response:
                json_str = judge_response.split("```json")[1].split("```")[0].strip()
            elif "```" in judge_response:
                json_str = judge_response.split("```")[1].split("```")[0].strip()
            else:
                json_str = judge_response.strip()

            rankings_data = json.loads(json_str)
            rankings = rankings_data["rankings"]

            # Sort by score (descending)
            rankings.sort(key=lambda x: x["score"], reverse=True)

            # Map back to candidates with scores; validate judge response fields and indices
            ranked_candidates = []
            for rank in rankings:
                if "candidate" not in rank or "score" not in rank:
                    raise ValueError(
                        f"Judge response missing required fields ('candidate', 'score'). "
                        f"Got: {rank}"
                    )
                candidate_idx = rank["candidate"] - 1  # Convert to 0-indexed
                if candidate_idx < 0 or candidate_idx >= len(candidates):
                    raise ValueError(
                        f"Judge returned invalid candidate index {rank['candidate']} "
                        f"(expected 1\u2013{len(candidates)})"
                    )
                ranked_candidates.append(
                    {
                        "response": candidates[candidate_idx],
                        "score": rank["score"],
                    }
                )

            return ranked_candidates

        return judge

    def judge_with_heuristic():
        """Factory returning a heuristic judge based on response length and quality indicators.

        This is a fallback when no external judge model is available.
        """

        def score_response(response: str) -> float:
            # Simple heuristic: longer responses with proper punctuation score higher
            score = 5.0  # Base score

            # Length bonus (up to 2 points)
            length_bonus = min(len(response) / 500.0, 2.0)
            score += length_bonus

            # Punctuation bonus
            if response.endswith((".", "!", "?")):
                score += 0.5

            # Multiple sentences bonus
            sentence_count = response.count(".") + response.count("!") + response.count("?")
            score += min(sentence_count * 0.2, 1.5)

            # Penalize very short responses
            if len(response) < 50:
                score -= 2.0

            return max(0.0, min(10.0, score))

        def judge(prompt: str, candidates: list[str], criteria: str) -> list[dict]:
            # Score all candidates
            scored = [
                {"response": candidate, "score": score_response(candidate)}
                for candidate in candidates
            ]

            # Sort by score (descending)
            scored.sort(key=lambda x: x["score"], reverse=True)

            return scored

        return judge

    # Parse model specification
    if model_spec.startswith("vllm://"):
        # Format: vllm://model-name@http://endpoint:port
        spec = model_spec.removeprefix("vllm://")
        if "@" not in spec:
            raise ValueError(
                f"Invalid vLLM spec '{model_spec}'. Expected format: "
                "'vllm://model-name@http://endpoint:port'"
            )
        model_name, endpoint = spec.split("@", 1)
        return judge_with_vllm(endpoint, model_name)

    elif model_spec.startswith("hf://"):
        # For HuggingFace models, use heuristic judge (can be extended later)
        return judge_with_heuristic()

    elif model_spec in ["heuristic", "simple"]:
        # Explicit heuristic judge
        return judge_with_heuristic()

    else:
        # Default: treat as OpenAI model
        return judge_with_openai(model_spec)
