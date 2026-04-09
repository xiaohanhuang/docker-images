"""LLM judge backend functions for calling evaluation APIs."""

import json
from typing import Optional


def _call_openai_judge(model: str, prompt: str, api_key: Optional[str] = None) -> dict:
    """Call OpenAI-compatible API for judging.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo")
        prompt: Evaluation prompt
        api_key: Optional API key (defaults to env var OPENAI_API_KEY)

    Returns:
        Parsed JSON response with score and reasoning
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert evaluator. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()
    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)


def _call_anthropic_judge(model: str, prompt: str, api_key: Optional[str] = None) -> dict:
    """Call Anthropic Claude API for judging.

    Args:
        model: Model name (e.g., "claude-3-5-sonnet-20241022")
        prompt: Evaluation prompt
        api_key: Optional API key (defaults to env var ANTHROPIC_API_KEY)

    Returns:
        Parsed JSON response with score and reasoning
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system="You are an expert evaluator. Respond only with JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    content = message.content[0].text.strip()
    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)


def _call_vllm_judge(endpoint: str, prompt: str, model: str = "default") -> dict:
    """Call vLLM or OpenAI-compatible endpoint for judging.

    Args:
        endpoint: vLLM endpoint URL (e.g., "http://localhost:8000/v1")
        prompt: Evaluation prompt
        model: Model identifier exposed by the vLLM endpoint

    Returns:
        Parsed JSON response with score and reasoning
    """
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key="not-required")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert evaluator. Respond only with JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content.strip()
    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)


def _build_evaluation_prompt(
    scorer: str, prediction: str, input_text: str, ground_truth: Optional[str], rubric: str
) -> str:
    """Build evaluation prompt for a specific scorer.

    Args:
        scorer: Name of the scoring dimension
        prediction: Model's predicted output
        input_text: Original input/question
        ground_truth: Optional ground truth answer
        rubric: Evaluation rubric

    Returns:
        Formatted prompt string
    """
    prompt = f"Evaluate the following answer using the {scorer} criteria:\n\n"
    prompt += f"Input/Question: {input_text}\n\n"
    prompt += f"Answer: {prediction}\n\n"
    if ground_truth:
        prompt += f"Reference Answer: {ground_truth}\n\n"
    prompt += f"Evaluation Criteria:\n{rubric}\n"
    return prompt
