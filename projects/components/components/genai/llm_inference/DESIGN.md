# Design Document: `genai.llm_inference` Component

**Component**: `genai.llm_inference`
**Category**: GenAI
**Image**: `genai-gpu`
**Status**: Implemented
**Author**: ML Platform Team
**Date**: 2026-03-03

---

## Executive Summary

The `llm_inference` component provides a unified interface for running inference on Large Language Models (LLMs) through multiple backends (local vLLM, OpenAI API, Anthropic API). This component is a critical building block for Act 3 of the AI-Native Platform Demo Walkthrough, enabling composable GenAI pipelines with minimal code.

### Key Features
- **Multi-Backend Support**: Seamlessly switch between vLLM (local GPU), OpenAI, and Anthropic
- **Batch Generation**: Process multiple prompts efficiently in a single task
- **Structured Output**: JSON mode for reliable structured data extraction
- **Cost Tracking**: Automatic token usage and cost estimation for API backends
- **Prompt Templates**: System prompt support for consistent behavior
- **Type Safety**: Full Flytekit integration with strong typing

---

## Design Principles

### 1. Consistency with Existing Components
Following patterns from `embeddings.py` and `rag.py`:
- Use `@task` decorator with explicit resource specifications
- Lazy imports for heavy dependencies (OpenAI, Anthropic, vLLM)
- Return structured Flyte types for serialization
- GPU resources requested only when needed (vLLM backend)

### 2. Backend Abstraction
The component provides a single unified API that abstracts three distinct backends:
- **vLLM** (local): GPU-accelerated inference using vLLM's OpenAI-compatible server
- **OpenAI**: Cloud API for GPT models
- **Anthropic**: Cloud API for Claude models

### 3. Flexibility vs. Simplicity
- **Simple path**: Single prompt → single output (auto-converted to list internally)
- **Batch path**: List of prompts → list of outputs
- **Configuration**: Sensible defaults with full control via parameters

### 4. Cost Transparency
For production workloads, knowing API costs is critical:
- Track input/output tokens for all backends
- Estimate costs using current API pricing
- Return metrics alongside results

---

## API Specification

### Task Signature

```python
@task(
    retries=2,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),  # GPU for vLLM
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
    """
    Run inference on any LLM (local via vLLM or remote via API).

    Args:
        model: Model ID (HuggingFace for vLLM) or API model name (gpt-4, claude-3-5-sonnet)
        prompts: Single prompt string or list of prompts for batch generation
        system_prompt: Optional system prompt to guide model behavior
        max_tokens: Maximum number of tokens to generate per prompt
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = random)
        top_p: Nucleus sampling threshold
        output_format: "text" for plain text, "json" for structured JSON output
        backend: "vllm" (local GPU), "openai" (API), or "anthropic" (API)

    Returns:
        Dictionary containing:
        - outputs: List of generated text strings (matches order of input prompts)
        - token_usage: Dict with "input_tokens" and "output_tokens" counts
        - cost: Estimated API cost in USD (0.0 for vLLM)

    Example:
        >>> result = llm_inference(
        ...     model="meta-llama/Llama-3.1-8B-Instruct",
        ...     prompts=["What is ML?", "Explain transformers"],
        ...     backend="vllm"
        ... )
        >>> print(result["outputs"][0])
        >>> print(f"Cost: ${result['cost']:.4f}")
    """
```

### Input Parameters

| Parameter | Type | Default | Constraints | Description |
|:----------|:-----|:--------|:------------|:------------|
| `model` | `str` | **required** | Non-empty string | HuggingFace model ID (vLLM) or API model name |
| `prompts` | `str` or `list[str]` | **required** | Non-empty | Single prompt or batch of prompts |
| `system_prompt` | `Optional[str]` | `None` | - | System-level instruction for the model |
| `max_tokens` | `int` | `512` | > 0, ≤ 4096 | Maximum tokens to generate per prompt |
| `temperature` | `float` | `0.7` | 0.0 - 2.0 | Sampling temperature |
| `top_p` | `float` | `0.9` | 0.0 - 1.0 | Nucleus sampling probability mass |
| `output_format` | `str` | `"text"` | "text" or "json" | Output format constraint |
| `backend` | `str` | `"vllm"` | "vllm", "openai", "anthropic" | Inference backend |

### Output Schema

```python
{
    "outputs": list[str],           # Generated texts (same order as input prompts)
    "token_usage": {
        "input_tokens": int,        # Total input tokens processed
        "output_tokens": int,       # Total output tokens generated
    },
    "cost": float,                  # Estimated cost in USD (0.0 for vLLM)
}
```

---

## Backend Implementation Details

### 1. vLLM Backend (Local GPU)

**Purpose**: High-throughput, low-latency inference on local GPUs using vLLM's optimized engine.

**Implementation Strategy**:
```python
# Lazy import inside task body
from openai import OpenAI

# vLLM runs OpenAI-compatible server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-required"
)

# For batch processing, iterate over prompts
outputs = []
for prompt in prompts:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,  # vLLM serves single model at /v1
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"} if output_format == "json" else None
    )
    outputs.append(response.choices[0].message.content)
```

**Assumptions**:
- vLLM server is pre-deployed or started as subprocess (like `vllm_deploy.py`)
- Model is already loaded in GPU memory
- Supports OpenAI Chat Completions API format

**Token Tracking**:
- Extract from `response.usage.prompt_tokens` and `response.usage.completion_tokens`
- vLLM accurately tracks tokens via tokenizer

**Cost**: Always `0.0` (local compute)

### 2. OpenAI Backend (Cloud API)

**Purpose**: Production-grade inference using OpenAI's GPT models.

**Implementation Strategy**:
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Batch processing with API rate limiting consideration
outputs = []
total_input_tokens = 0
total_output_tokens = 0

for prompt in prompts:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,  # e.g., "gpt-4o", "gpt-4-turbo"
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_object"} if output_format == "json" else None
    )

    outputs.append(response.choices[0].message.content)
    total_input_tokens += response.usage.prompt_tokens
    total_output_tokens += response.usage.completion_tokens
```

**API Key Management**:
- Read from environment variable `OPENAI_API_KEY`
- Fail fast with clear error if not set
- Document in error message: "Set OPENAI_API_KEY environment variable"

**Token Tracking**:
- OpenAI API returns exact token counts in `response.usage`

**Cost Estimation** (as of 2026):
```python
# Pricing per 1M tokens (update as needed)
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = OPENAI_PRICING.get(model, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

### 3. Anthropic Backend (Cloud API)

**Purpose**: Access to Claude models for specific capabilities (long context, strong reasoning).

**Implementation Strategy**:
```python
from anthropic import Anthropic
import os

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

outputs = []
total_input_tokens = 0
total_output_tokens = 0

for prompt in prompts:
    messages = [{"role": "user", "content": prompt}]

    # Anthropic uses separate system parameter
    response = client.messages.create(
        model=model,  # e.g., "claude-3-5-sonnet-20241022"
        system=system_prompt if system_prompt else None,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # Extract text from response
    content = response.content[0].text
    outputs.append(content)

    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens
```

**API Key Management**:
- Read from environment variable `ANTHROPIC_API_KEY`

**JSON Mode**:
- Anthropic doesn't have native JSON mode
- Use prompt engineering: append to system_prompt "You must respond with valid JSON only."

**Cost Estimation** (as of 2026):
```python
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}
```

---

## Resource Management

### GPU Requirements

**vLLM Backend**:
- Requires GPU: `requests=Resources(gpu="1")` for 7B-13B models
- Larger models (70B+) may require multi-GPU: `gpu="4"` or `gpu="8"`
- Memory: Minimum 16Gi RAM, 32Gi for larger models

**OpenAI/Anthropic Backends**:
- No GPU required: `requests=Resources(cpu="4", mem="8Gi")`
- CPU-only task reduces cluster cost
- Can use `data-cpu` image instead of `genai-gpu`

**Dynamic Resource Selection**:
```python
# Pseudo-code for future optimization
def get_resources(backend: str) -> Resources:
    if backend == "vllm":
        return Resources(cpu="4", mem="16Gi", gpu="1")
    else:
        return Resources(cpu="4", mem="8Gi")  # API backends don't need GPU
```

### Caching Strategy

**Cache Key Components**:
- Function signature (implicit)
- All input parameters: `model`, `prompts`, `system_prompt`, etc.
- `cache_version="1.0"` for breaking changes

**When to Use Cache**:
- ✅ Deterministic generation (temperature=0.0)
- ✅ Reproducible evaluation benchmarks
- ✅ Repeated queries (e.g., prompt engineering iterations)
- ❌ Creative generation with high temperature
- ❌ Real-time user-facing applications

**Cache Invalidation**:
- Increment `cache_version` when:
  - Implementation logic changes
  - Backend behavior changes (e.g., API version updates)
  - Output format changes

---

## Error Handling & Edge Cases

### Input Validation

```python
def validate_inputs(
    model: str,
    prompts: Union[str, list[str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_format: str,
    backend: str,
) -> None:
    """Validate all inputs before backend dispatch."""

    # Model validation
    if not model or not model.strip():
        raise ValueError("model must be a non-empty string")

    # Prompts validation
    if isinstance(prompts, str):
        if not prompts.strip():
            raise ValueError("prompt string cannot be empty")
    elif isinstance(prompts, list):
        if len(prompts) == 0:
            raise ValueError("prompts list cannot be empty")
        if any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("all prompts must be non-empty strings")
    else:
        raise TypeError("prompts must be str or list[str]")

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
```

### Backend-Specific Errors

**vLLM**:
- `ConnectionError`: vLLM server not running or unreachable
  - Error message: "vLLM server not reachable at http://localhost:8000. Ensure vLLM is deployed."
- `RuntimeError`: Model not loaded or OOM
  - Error message: "vLLM failed to process request. Check model is loaded and GPU memory is sufficient."

**OpenAI**:
- `AuthenticationError`: Invalid API key
  - Error message: "OpenAI API key invalid. Set OPENAI_API_KEY environment variable."
- `RateLimitError`: API rate limit exceeded
  - Error message: "OpenAI rate limit exceeded. Implement retry logic or reduce request rate."
- `InvalidRequestError`: Invalid model name or parameters
  - Error message: "Invalid model or parameters for OpenAI API: {details}"

**Anthropic**:
- `AuthenticationError`: Invalid API key
  - Error message: "Anthropic API key invalid. Set ANTHROPIC_API_KEY environment variable."
- Similar rate limit and validation errors as OpenAI

### Retry Strategy

Flyte task decorator includes `retries=2` for transient failures:
- Network timeouts
- API rate limiting (with exponential backoff)
- Temporary service unavailability

Non-retryable errors (fail fast):
- Invalid API keys
- Validation errors
- Unsupported model names

---

## Testing Strategy

### Unit Tests

**File**: `tests/components/genai/test_llm_inference.py`

```python
import pytest
from projects.components.components.genai.llm_inference import llm_inference

class TestLLMInference:
    """Unit tests for llm_inference component."""

    def test_input_validation_empty_model(self):
        """Test that empty model raises ValueError."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            llm_inference(model="", prompts="test", backend="openai")

    def test_input_validation_empty_prompts(self):
        """Test that empty prompts raise ValueError."""
        with pytest.raises(ValueError, match="prompt string cannot be empty"):
            llm_inference(model="gpt-4", prompts="", backend="openai")

    def test_input_validation_invalid_temperature(self):
        """Test that invalid temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be between"):
            llm_inference(model="gpt-4", prompts="test", temperature=3.0, backend="openai")

    def test_input_validation_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            llm_inference(model="gpt-4", prompts="test", backend="invalid")

    def test_single_prompt_converted_to_list(self):
        """Test that single string prompt is converted to list internally."""
        # This would be a mock test - requires mocking OpenAI client
        pass

    def test_batch_prompts_preserves_order(self):
        """Test that batch output order matches input order."""
        # Mock test to verify outputs align with input prompts
        pass
```

### Integration Tests

**Requirements**:
- Mock API clients (OpenAI, Anthropic) to avoid real API calls
- Use `pytest-mock` or `unittest.mock`
- Test each backend separately

**Test Scenarios**:
1. **vLLM Backend**: Mock vLLM server response
2. **OpenAI Backend**: Mock OpenAI client with realistic response
3. **Anthropic Backend**: Mock Anthropic client
4. **Token Counting**: Verify token usage aggregation
5. **Cost Calculation**: Verify pricing logic
6. **JSON Mode**: Verify response_format parameter passed correctly

### End-to-End Tests

**Manual Testing Checklist**:
- [ ] Deploy vLLM server with a 7B model (e.g., Llama-3.1-8B)
- [ ] Run `llm_inference` with vLLM backend locally
- [ ] Verify outputs are generated correctly
- [ ] Verify token counts are accurate
- [ ] Test with OpenAI API (requires API key)
- [ ] Test with Anthropic API (requires API key)
- [ ] Test batch processing with 10+ prompts
- [ ] Test JSON mode output parsing

---

## Integration with Existing Components

### Composition Example: RAG Pipeline

```python
from flytekit import workflow
from projects.components.components.genai.embeddings import generate_embeddings
from projects.components.components.genai.llm_inference import llm_inference

@workflow
def enhanced_rag_pipeline(
    query: str,
    corpus_file: FlyteFile,
    top_k: int = 5,
) -> str:
    """RAG pipeline using component library."""

    # Step 1: Generate embeddings for corpus
    embeddings = generate_embeddings(
        texts=corpus_file,
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Step 2: Retrieve relevant documents (simplified)
    # In real implementation, would include vector search logic

    # Step 3: Generate answer using LLM
    context_prompt = f"Context: {{retrieved_docs}}\n\nQuestion: {query}"
    result = llm_inference(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompts=context_prompt,
        system_prompt="Answer based only on the provided context.",
        backend="vllm"
    )

    return result["outputs"][0]
```

### Usage in Demo Walkthrough (Act 3)

```python
from ml_platform_sdk.genai import text_chunker, vector_indexer, llm_inference

@workflow
def build_rag():
    chunks = text_chunker(data="s3://docs")
    index = vector_indexer(chunks)

    # Query the index
    answer = llm_inference(
        model="meta-llama/Llama-3.1-8B-Instruct",
        prompts="What is the platform's architecture?",
        backend="vllm"
    )

    return answer
```

---

## Future Enhancements

### Phase 1: Streaming Support
```python
def llm_inference_stream(
    model: str,
    prompt: str,
    backend: str = "vllm"
) -> Iterator[str]:
    """Stream tokens as they are generated."""
    # Useful for real-time chat applications
    # Requires async task support in Flyte
```

### Phase 2: Prompt Templates
```python
def llm_inference_with_template(
    template: str,  # e.g., "Summarize: {text}"
    template_vars: dict[str, str],
    model: str,
    backend: str = "vllm"
) -> dict[str, Any]:
    """Use Jinja2 templates for complex prompts."""
```

### Phase 3: Multi-Model Ensemble
```python
def llm_inference_ensemble(
    prompts: list[str],
    models: list[str],  # e.g., ["gpt-4", "claude-3-opus"]
    voting_strategy: str = "majority"
) -> dict[str, Any]:
    """Run inference on multiple models and aggregate results."""
```

### Phase 4: Function Calling Support
```python
def llm_inference_with_tools(
    prompt: str,
    tools: list[dict],  # OpenAI function calling schema
    model: str,
    backend: str = "openai"
) -> dict[str, Any]:
    """Enable function calling for agentic workflows."""
```

---

## Deployment Considerations

### Docker Image Requirements

**Image**: `genai-gpu`

**Required Dependencies** (already in Dockerfile):
- `vllm==0.5.0`
- `openai==1.34.0`
- Base: `anthropic` (needs to be added)

**Dockerfile Update Required**:
```dockerfile
# Add to projects/components/images/genai-gpu/Dockerfile
RUN pip install --no-cache-dir \
    anthropic==0.25.0  # Add Anthropic SDK
```

### Environment Variables

**Required for API Backends**:
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Kubernetes Secret Management**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-api-keys
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
  ANTHROPIC_API_KEY: "sk-ant-..."
```

**Flyte Task Injection**:
```python
@task(
    secret_requests=[
        Secret(key="OPENAI_API_KEY", group="llm-api-keys"),
        Secret(key="ANTHROPIC_API_KEY", group="llm-api-keys"),
    ]
)
def llm_inference(...):
    ...
```

### vLLM Server Deployment

**Option 1**: Pre-deployed vLLM Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-8b
spec:
  selector:
    app: vllm
  ports:
    - port: 8000
      targetPort: 8000
```

**Option 2**: Ephemeral vLLM (within task)
- Use `vllm_deploy.py` as a prerequisite task
- Pass endpoint URL to `llm_inference`

**Recommended**: Pre-deployed for Act 3 demo (faster, more reliable)

---

## Performance Benchmarks (Expected)

| Backend | Model | Batch Size | Throughput (tok/s) | Latency (s) | Cost/1K prompts |
|:--------|:------|:-----------|:-------------------|:------------|:----------------|
| vLLM | Llama-3.1-8B | 1 | ~100 | 2-3 | $0 (local) |
| vLLM | Llama-3.1-8B | 10 | ~80 | 5-8 | $0 (local) |
| OpenAI | GPT-4o | 1 | ~50 | 1-2 | $0.50 |
| OpenAI | GPT-4o | 10 | ~50 | 10-15 | $5.00 |
| Anthropic | Claude-3.5-Sonnet | 1 | ~60 | 1-2 | $0.75 |

**Notes**:
- vLLM throughput depends on GPU type (A100 > V100)
- API latency includes network overhead
- Batch processing on APIs processes serially (no batch API used)

---

## Documentation & Examples

### Docstring Format

Follow Google-style docstrings (consistent with `embeddings.py` and `rag.py`):
```python
def llm_inference(...) -> dict[str, Any]:
    """Run inference on any LLM (local via vLLM or remote via API).

    Supports batch generation, structured output (JSON mode), and multiple
    backends. Automatically tracks token usage and estimates API costs.

    Args:
        model: Model ID (HuggingFace) or API endpoint URL.
        prompts: Single prompt string or list of prompts for batch generation.
        ...

    Returns:
        Dictionary containing:
        - outputs: List of generated text strings
        - token_usage: Input/output token counts
        - cost: Estimated API cost (0.0 for vLLM)

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
        AuthenticationError: If API key is invalid (openai/anthropic backends)
    """
```

### Usage Examples in README

Add to `projects/components/README.md`:

```markdown
### GenAI: LLM Inference

Run inference on any LLM (local or remote) with a unified API.

**vLLM (Local GPU)**:
```python
from components.genai.llm_inference import llm_inference

result = llm_inference(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompts=["What is ML?", "Explain transformers"],
    backend="vllm"
)
print(result["outputs"])  # ['ML is...', 'Transformers are...']
print(f"Cost: ${result['cost']}")  # $0.00
```

**OpenAI API**:
```python
result = llm_inference(
    model="gpt-4o",
    prompts="Write a haiku about ML",
    temperature=0.9,
    backend="openai"
)
```

**Anthropic API with JSON Mode**:
```python
result = llm_inference(
    model="claude-3-5-sonnet-20241022",
    prompts="Extract entities from: 'Apple announced iPhone in 2007'",
    system_prompt="Respond with JSON: {company, product, year}",
    output_format="json",
    backend="anthropic"
)
```
```

---

## Security & Privacy

### API Key Management
- **Never** hardcode API keys in code
- Use Kubernetes Secrets + Flyte's `Secret` integration
- Rotate keys regularly (90-day policy recommended)

### Data Privacy
- Prompts sent to OpenAI/Anthropic APIs may be logged by providers
- For sensitive data, use vLLM backend (local, no data leaves cluster)
- Check API provider's data retention policies

### Rate Limiting
- Implement exponential backoff for API errors
- Consider API usage quotas in production
- Monitor costs with alerts (CloudWatch/Prometheus)

---

## Success Criteria

This design is considered complete when:
- ✅ Design document reviewed and approved
- ⬜ Implementation matches API specification
- ⬜ All three backends (vLLM, OpenAI, Anthropic) functional
- ⬜ Unit tests pass with >80% coverage
- ⬜ Integration tests pass with mocked APIs
- ⬜ End-to-end test with real vLLM deployment succeeds
- ⬜ Cost tracking verified against API billing
- ⬜ Documentation complete with usage examples
- ⬜ Component registered in Flyte and discoverable via CLI

---

## References

### Internal
- `projects/components/components/genai/embeddings.py` - Component pattern reference
- `projects/components/components/genai/rag.py` - OpenAI client usage pattern
- `projects/components/components/serving/vllm_deploy.py` - vLLM deployment pattern
- `demo_walkthrough.md` - Act 3 requirements

### External
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference)
- [Flytekit Documentation](https://docs.flyte.org/)

---

## Appendix: Alternative Design Considerations

### Why Not Use LangChain?
**Decision**: Implement direct API clients instead of LangChain.

**Rationale**:
- Simpler dependency graph (LangChain is heavy)
- More control over error handling
- Easier to optimize for batch processing
- LangChain can be added as a separate component if needed

### Why Not Use OpenAI's Batch API?
**Decision**: Process prompts serially for now.

**Rationale**:
- Batch API has 24-hour turnaround time (too slow for demo)
- Serial processing is simpler and more predictable
- Can add async batch processing in Phase 2

### Why Not Support Streaming?
**Decision**: Deferred to Phase 1 enhancement.

**Rationale**:
- Flyte tasks are synchronous by default
- Streaming requires async task support (complex)
- Batch generation is more important for Act 3 demo

---

**End of Design Document**

*This document will be updated as implementation proceeds and feedback is incorporated.*
