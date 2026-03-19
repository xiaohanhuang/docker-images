# Implementation Notes: `llm_inference` Component

**Quick Reference for Implementers**

---

## File Location

```
projects/components/components/genai/llm_inference.py
```

---

## Implementation Checklist

### Phase 1: Core Structure
- [ ] Create `llm_inference.py` in `projects/components/components/genai/`
- [ ] Add module docstring with image tag and description
- [ ] Import necessary Flytekit types
- [ ] Define task decorator with resource requirements
- [ ] Define function signature matching design spec

### Phase 2: Input Validation
- [ ] Implement `validate_inputs()` helper function
- [ ] Add checks for all parameter constraints
- [ ] Provide clear error messages for each validation failure
- [ ] Test validation with edge cases

### Phase 3: Backend Implementation
- [ ] Implement vLLM backend logic
  - [ ] OpenAI client with localhost endpoint
  - [ ] Message formatting (system + user)
  - [ ] Response extraction
  - [ ] Token counting
- [ ] Implement OpenAI backend logic
  - [ ] API key from environment
  - [ ] Proper error handling
  - [ ] Response format support (text/json)
  - [ ] Cost calculation
- [ ] Implement Anthropic backend logic
  - [ ] API key from environment
  - [ ] System prompt handling (separate param)
  - [ ] JSON mode via prompt engineering
  - [ ] Cost calculation

### Phase 4: Response Handling
- [ ] Aggregate outputs from all prompts
- [ ] Calculate total token usage
- [ ] Estimate costs for API backends
- [ ] Return structured dictionary

### Phase 5: Error Handling
- [ ] Wrap backend calls in try/except
- [ ] Handle API authentication errors
- [ ] Handle rate limit errors
- [ ] Handle connection errors (vLLM)
- [ ] Provide actionable error messages

### Phase 6: Testing
- [ ] Write unit tests for validation
- [ ] Write integration tests with mocked APIs
- [ ] Test each backend separately
- [ ] Test batch processing
- [ ] Test JSON mode
- [ ] Test cost calculation accuracy

### Phase 7: Documentation
- [ ] Add comprehensive docstring
- [ ] Add usage examples in docstring
- [ ] Update `projects/components/README.md`
- [ ] Add to component registry (if exists)

### Phase 8: Docker Image
- [ ] Update `genai-gpu/Dockerfile` to include Anthropic SDK
- [ ] Rebuild and test image
- [ ] Push to ECR

---

## Code Template (Skeleton)

```python
"""
GenAI — configurable LLM inference with multi-backend support.

Image: genai-gpu
"""

from typing import Any, Optional, Union

from flytekit import Resources, task


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
) -> dict[str, Any]:
    """Run inference on any LLM (local via vLLM or remote via API).

    Args:
        model: Model ID (HuggingFace for vLLM) or API model name.
        prompts: Single prompt string or list of prompts.
        system_prompt: Optional system prompt.
        max_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        output_format: "text" or "json".
        backend: "vllm", "openai", or "anthropic".

    Returns:
        Dictionary with keys: outputs, token_usage, cost.
    """
    # Import inside task body (lazy loading)
    import os

    # Normalize prompts to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # Validate inputs
    _validate_inputs(model, prompts, max_tokens, temperature, top_p, output_format, backend)

    # Dispatch to backend
    if backend == "vllm":
        return _run_vllm(model, prompts, system_prompt, max_tokens, temperature, top_p, output_format)
    elif backend == "openai":
        return _run_openai(model, prompts, system_prompt, max_tokens, temperature, top_p, output_format)
    elif backend == "anthropic":
        return _run_anthropic(model, prompts, system_prompt, max_tokens, temperature, top_p, output_format)


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
    # TODO: Implement validation logic
    pass


def _run_vllm(
    model: str,
    prompts: list[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
) -> dict[str, Any]:
    """Execute inference using vLLM backend."""
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")

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
                response_format={"type": "json_object"} if output_format == "json" else None,
            )
            outputs.append(response.choices[0].message.content)
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
        except Exception as e:
            raise RuntimeError(f"vLLM inference failed: {e}")

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
                response_format={"type": "json_object"} if output_format == "json" else None,
            )
            outputs.append(response.choices[0].message.content)
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {e}")

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

    from anthropic import Anthropic

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
            raise RuntimeError(f"Anthropic API request failed: {e}")

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
    """Calculate estimated cost for OpenAI API usage."""
    # Pricing per 1M tokens (as of 2026)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def _calculate_anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost for Anthropic API usage."""
    # Pricing per 1M tokens (as of 2026)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

---

## Testing Strategy

### Unit Tests

Create `tests/components/genai/test_llm_inference.py`:

```python
import pytest
from unittest.mock import Mock, patch
from projects.components.components.genai.llm_inference import llm_inference


class TestInputValidation:
    """Test input validation logic."""

    def test_empty_model_raises_error(self):
        with pytest.raises(ValueError, match="model"):
            llm_inference(model="", prompts="test", backend="vllm")

    def test_empty_prompts_raises_error(self):
        with pytest.raises(ValueError, match="prompts"):
            llm_inference(model="test", prompts="", backend="vllm")

    def test_invalid_temperature_raises_error(self):
        with pytest.raises(ValueError, match="temperature"):
            llm_inference(model="test", prompts="test", temperature=3.0, backend="vllm")

    def test_invalid_backend_raises_error(self):
        with pytest.raises(ValueError, match="backend"):
            llm_inference(model="test", prompts="test", backend="invalid")


class TestVLLMBackend:
    """Test vLLM backend logic."""

    @patch("openai.OpenAI")
    def test_single_prompt_vllm(self, mock_openai):
        """Test vLLM with single prompt."""
        # Mock OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test output"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = llm_inference(model="test-model", prompts="test prompt", backend="vllm")

        assert result["outputs"] == ["Test output"]
        assert result["token_usage"]["input_tokens"] == 10
        assert result["token_usage"]["output_tokens"] == 20
        assert result["cost"] == 0.0

    @patch("openai.OpenAI")
    def test_batch_prompts_vllm(self, mock_openai):
        """Test vLLM with batch prompts."""
        # Mock multiple responses
        mock_openai.return_value.chat.completions.create.side_effect = [
            Mock(
                choices=[Mock(message=Mock(content="Output 1"))],
                usage=Mock(prompt_tokens=10, completion_tokens=20),
            ),
            Mock(
                choices=[Mock(message=Mock(content="Output 2"))],
                usage=Mock(prompt_tokens=15, completion_tokens=25),
            ),
        ]

        result = llm_inference(
            model="test-model", prompts=["prompt 1", "prompt 2"], backend="vllm"
        )

        assert len(result["outputs"]) == 2
        assert result["token_usage"]["input_tokens"] == 25
        assert result["token_usage"]["output_tokens"] == 45


class TestOpenAIBackend:
    """Test OpenAI backend logic."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_openai_with_api_key(self, mock_openai):
        """Test OpenAI backend with valid API key."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test output"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=200)
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = llm_inference(model="gpt-4o", prompts="test", backend="openai")

        assert result["outputs"] == ["Test output"]
        assert result["cost"] > 0  # Should have calculated cost

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_without_api_key_raises_error(self):
        """Test that missing API key raises clear error."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            llm_inference(model="gpt-4", prompts="test", backend="openai")


class TestAnthropicBackend:
    """Test Anthropic backend logic."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("anthropic.Anthropic")
    def test_anthropic_with_api_key(self, mock_anthropic):
        """Test Anthropic backend with valid API key."""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test output")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=200)
        mock_anthropic.return_value.messages.create.return_value = mock_response

        result = llm_inference(
            model="claude-3-5-sonnet-20241022", prompts="test", backend="anthropic"
        )

        assert result["outputs"] == ["Test output"]
        assert result["cost"] > 0


class TestCostCalculation:
    """Test cost calculation accuracy."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("openai.OpenAI")
    def test_openai_cost_calculation(self, mock_openai):
        """Test that OpenAI cost is calculated correctly."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test"))]
        mock_response.usage = Mock(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = llm_inference(model="gpt-4o", prompts="test", backend="openai")

        # For gpt-4o: input=$2.50/1M, output=$10.00/1M
        expected_cost = 2.50 + 10.00
        assert abs(result["cost"] - expected_cost) < 0.01
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Import Errors at Registration
**Problem**: Importing heavy dependencies (OpenAI, Anthropic) at module level causes slow registration.

**Solution**: Import inside task body (lazy loading):
```python
def llm_inference(...):
    # Import here, not at top of file
    from openai import OpenAI
    from anthropic import Anthropic
```

### Pitfall 2: API Key Not Available in Container
**Problem**: Environment variable set locally but not in Flyte task.

**Solution**: Use Flyte Secrets:
```python
from flytekit import Secret

@task(
    secret_requests=[
        Secret(key="OPENAI_API_KEY", group="llm-api-keys"),
    ]
)
def llm_inference(...):
    ...
```

### Pitfall 3: vLLM Server Not Running
**Problem**: Task tries to connect to localhost:8000 but vLLM isn't deployed.

**Solution**: Add clear error message:
```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    raise RuntimeError(
        f"vLLM server not reachable. Ensure vLLM is deployed and accessible at "
        f"http://localhost:8000. Original error: {e}"
    )
```

### Pitfall 4: JSON Mode Not Working on Anthropic
**Problem**: Anthropic doesn't have native JSON mode like OpenAI.

**Solution**: Use prompt engineering:
```python
if output_format == "json":
    system_prompt += "\nYou must respond with valid JSON only."
```

### Pitfall 5: Cost Estimation Outdated
**Problem**: API pricing changes but our constants are stale.

**Solution**:
- Add comment with last updated date
- Log warning if model not in pricing table
- Return 0.0 for unknown models (don't fail)

---

## Integration Testing Locally

### Test vLLM Backend

```bash
# 1. Start vLLM server (requires GPU)
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct

# 2. Test in Python
python -c "
from projects.components.components.genai.llm_inference import llm_inference
result = llm_inference(
    model='meta-llama/Llama-3.1-8B-Instruct',
    prompts='What is 2+2?',
    backend='vllm'
)
print(result)
"
```

### Test OpenAI Backend

```bash
# 1. Set API key
export OPENAI_API_KEY="sk-..."

# 2. Test in Python
python -c "
from projects.components.components.genai.llm_inference import llm_inference
result = llm_inference(
    model='gpt-4o',
    prompts='What is 2+2?',
    backend='openai'
)
print(result)
print(f'Cost: ${result[\"cost\"]:.4f}')
"
```

### Test Anthropic Backend

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Test in Python
python -c "
from projects.components.components.genai.llm_inference import llm_inference
result = llm_inference(
    model='claude-3-5-sonnet-20241022',
    prompts='What is 2+2?',
    backend='anthropic'
)
print(result)
print(f'Cost: ${result[\"cost\"]:.4f}')
"
```

---

## Deployment Checklist

- [ ] Update `genai-gpu/Dockerfile` to include Anthropic SDK
- [ ] Rebuild Docker image
- [ ] Push to ECR
- [ ] Create Kubernetes Secret for API keys
- [ ] Register component in Flyte: `pyflyte register projects/components/components/genai/llm_inference.py`
- [ ] Verify component appears in CLI: `ml-plat component list`
- [ ] Test end-to-end with real Flyte execution

---

## Performance Optimization Tips

### For vLLM Backend
- Use `--tensor-parallel-size` for multi-GPU inference
- Set appropriate `--max-model-len` to avoid OOM
- Use `--gpu-memory-utilization 0.9` to maximize throughput

### For API Backends
- Consider async/await for parallel API calls (future enhancement)
- Implement exponential backoff for rate limits
- Cache responses when temperature=0 (deterministic)

### For Batch Processing
- Current: Serial processing (simple, predictable)
- Future: Parallel API calls using `asyncio` (faster but complex)

---

## Maintenance & Updates

### When to Update Pricing
- Check OpenAI/Anthropic pricing pages quarterly
- Update constants in `_calculate_*_cost()` functions
- Add new models as they are released

### When to Bump Cache Version
- Implementation logic changes
- Backend behavior changes (e.g., API updates)
- Output format changes

### When to Add New Backend
- Follow the pattern: `_run_<backend>()` helper function
- Add to `backend` parameter validation
- Add cost calculation function
- Update tests
- Update documentation

---

**End of Implementation Notes**
