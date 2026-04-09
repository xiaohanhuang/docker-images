"""LLM inference backends (vLLM, OpenAI, Anthropic) and cost calculation."""

from typing import Any, Optional


def _validate_inputs(
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
    backend: str,
) -> None:
    """Validate all inputs before processing."""
    # Model validation
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")

    # Prompts validation
    if len(prompts) == 0:
        raise ValueError("prompts list cannot be empty")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("all prompts must be non-empty strings")

    # Parameter validation
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")

    # Format validation
    if output_format not in ["text", "json"]:
        raise ValueError("output_format must be 'text' or 'json'")

    # Backend validation
    if backend not in ["vllm", "openai", "anthropic"]:
        raise ValueError("backend must be 'vllm', 'openai', or 'anthropic'")


def _run_vllm(
    model: str,
    prompts: list[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
    vllm_endpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Execute inference using vLLM backend."""
    import os

    from openai import OpenAI

    endpoint = vllm_endpoint or os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
    client = OpenAI(base_url=endpoint, api_key="not-required")

    outputs = []
    total_input_tokens = 0
    total_output_tokens = 0

    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=(
                    {"type": "json_object"} if output_format == "json" else {"type": "text"}
                ),
            )
            outputs.append(response.choices[0].message.content)
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
        except Exception as e:
            raise RuntimeError(
                f"vLLM inference failed. Ensure vLLM is deployed and accessible at "
                f"{endpoint}. Original error: {e}"
            ) from e

    return {
        "outputs": outputs,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "cost": 0.0,  # vLLM is local, no cost
    }


def _run_openai(
    model: str,
    prompts: list[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
) -> dict[str, Any]:
    """Execute inference using OpenAI API."""
    import os

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key)

    outputs = []
    total_input_tokens = 0
    total_output_tokens = 0

    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=(
                    {"type": "json_object"} if output_format == "json" else {"type": "text"}
                ),
            )
            outputs.append(response.choices[0].message.content)
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {e}") from e

    # Calculate cost
    cost = _calculate_openai_cost(model, total_input_tokens, total_output_tokens)

    return {
        "outputs": outputs,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "cost": cost,
    }


def _run_anthropic(
    model: str,
    prompts: list[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
) -> dict[str, Any]:
    """Execute inference using Anthropic API."""
    import os

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for the Anthropic backend. "
            "Install it with: pip install anthropic>=0.25.0"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = Anthropic(api_key=api_key)

    # Handle JSON mode via prompt engineering
    effective_system_prompt = system_prompt or ""
    if output_format == "json":
        effective_system_prompt += "\nYou must respond with valid JSON only."

    outputs = []
    total_input_tokens = 0
    total_output_tokens = 0

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]

        try:
            response = client.messages.create(
                model=model,
                system=effective_system_prompt if effective_system_prompt else None,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            content = response.content[0].text
            outputs.append(content)
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
        except Exception as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e

    # Calculate cost
    cost = _calculate_anthropic_cost(model, total_input_tokens, total_output_tokens)

    return {
        "outputs": outputs,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
        "cost": cost,
    }


def _calculate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for OpenAI API usage.

    Pricing per 1M tokens (as of 2026-03-03).
    """
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    price = pricing.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]
    return input_cost + output_cost


def _calculate_anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for Anthropic API usage.

    Pricing per 1M tokens (as of 2026-03-03).
    """
    pricing = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    price = pricing.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * price["input"]
    output_cost = (output_tokens / 1_000_000) * price["output"]
    return input_cost + output_cost
