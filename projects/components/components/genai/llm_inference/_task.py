"""Flyte task definition for llm_inference component."""

from typing import Any, Optional, Union

from flytekit import Resources, task

from ._backends import _validate_inputs


@task(
    retries=2,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def llm_inference(
    model: str,
    prompts: Union[str, list[str]],
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    output_format: str = "text",
    backend: str = "vllm",
    vllm_endpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Run inference on any LLM (local via vLLM or remote via API).

    Supports batch generation, structured output (JSON mode), and multiple
    backends. Automatically tracks token usage and estimates API costs.

    Args:
        model: Model ID (HuggingFace) or API model name (gpt-4, claude-3-5-sonnet).
        prompts: Single prompt string or list of prompts for batch generation.
        system_prompt: Optional system prompt to guide model behavior.
        max_tokens: Maximum number of tokens to generate per prompt.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = random).
        top_p: Nucleus sampling threshold (0.0-1.0).
        output_format: "text" for plain text, "json" for structured JSON output.
        backend: "vllm" (local GPU), "openai" (API), or "anthropic" (API).
        vllm_endpoint: Base URL for the vLLM server. Defaults to the
            ``VLLM_ENDPOINT`` environment variable or
            ``http://localhost:8000/v1`` when not set. Override this to
            target a Kubernetes Service when vLLM is deployed separately.

    Returns:
        Dictionary containing:
        - outputs: List of generated text strings (matches order of input prompts)
        - token_usage: Dict with "input_tokens" and "output_tokens" counts
        - cost: Estimated API cost in USD (0.0 for vLLM)

    Example:
        >>> result = llm_inference(
        ...     model="meta-llama/Llama-3.1-8B-Instruct",
        ...     prompts="Explain quantum computing in simple terms",
        ...     backend="vllm"
        ... )
        >>> print(result["outputs"][0])
        >>> print(f"Tokens used: {result['token_usage']}")

    Raises:
        ValueError: If inputs are invalid (empty model, invalid backend, etc.)
        ConnectionError: If vLLM server is unreachable (vllm backend only)
        RuntimeError: If API request fails (authentication, rate limit, etc.)
    """
    # Normalize prompts to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # Lazy import backend functions
    from ._backends import _run_anthropic, _run_openai, _run_vllm

    # Validate inputs
    _validate_inputs(model, prompts, max_tokens, temperature, top_p, output_format, backend)

    # Dispatch to backend
    if backend == "vllm":
        return _run_vllm(
            model,
            prompts,
            system_prompt,
            max_tokens,
            temperature,
            top_p,
            output_format,
            vllm_endpoint,
        )
    elif backend == "openai":
        return _run_openai(
            model, prompts, system_prompt, max_tokens, temperature, top_p, output_format
        )
    elif backend == "anthropic":
        return _run_anthropic(
            model, prompts, system_prompt, max_tokens, temperature, top_p, output_format
        )
    else:
        raise AssertionError(f"Unhandled backend '{backend}' after validation")
