# LLM Inference Component - Design Summary

**Component**: `genai.llm_inference`
**Status**: 🚧 Implementation In Progress (Design Complete)
**Date**: 2026-03-03
**Next Phase**: Testing & Integration

---

## Overview

This design package provides complete specifications for implementing the `llm_inference` component, a critical building block for Act 3 of the AI-Native Platform Demo Walkthrough.

## Design Deliverables

### 1. Main Design Document
**File**: `llm_inference_design.md`
**Size**: 867 lines, 26 KB

**Contents**:
- Executive summary and key features
- Complete API specification with all input/output types
- Detailed backend implementations (vLLM, OpenAI, Anthropic)
- Resource management and caching strategy
- Error handling and edge cases
- Testing strategy with example tests
- Integration with existing components
- Security and privacy considerations
- Performance benchmarks
- Future enhancements roadmap

### 2. Implementation Guide
**File**: `IMPLEMENTATION_NOTES.md`
**Size**: 682 lines, 21 KB

**Contents**:
- Step-by-step implementation checklist
- Complete code template with skeleton functions
- Unit test examples with pytest
- Common pitfalls and solutions
- Local testing instructions for each backend
- Deployment checklist
- Performance optimization tips
- Maintenance guidelines

---

## Key Design Decisions

### 1. Multi-Backend Architecture
**Decision**: Support three backends through a unified API.

**Rationale**:
- **vLLM (local)**: High throughput, zero cost, full control
- **OpenAI (cloud)**: Production-grade, wide model selection
- **Anthropic (cloud)**: Long context, strong reasoning (Claude)

**Implementation**: Backend dispatch pattern with separate helper functions.

### 2. Batch Processing
**Decision**: Accept both single string and list of strings as input.

**Rationale**:
- Simplicity for single-prompt use cases
- Efficiency for batch workloads
- Automatic normalization to list internally

**Implementation**: `prompts: Union[str, list[str]]`

### 3. Cost Transparency
**Decision**: Always return token usage and estimated cost.

**Rationale**:
- Production deployments need cost visibility
- Enables budget tracking and optimization
- Helps users make informed backend choices

**Implementation**: Return `{"outputs": [...], "token_usage": {...}, "cost": float}`

### 4. Lazy Imports
**Decision**: Import heavy dependencies inside task body, not at module level.

**Rationale**:
- Fast Flyte registration (don't load PyTorch/API clients during registration)
- Follows existing pattern in `embeddings.py` and `rag.py`
- Reduces memory footprint

**Implementation**: All `import openai`, `import anthropic` inside task function.

### 5. JSON Mode via Response Format
**Decision**: Use native `response_format` for OpenAI, prompt engineering for Anthropic.

**Rationale**:
- OpenAI has native JSON mode (reliable)
- Anthropic requires prompt engineering (acceptable tradeoff)
- Unified parameter interface for users

**Implementation**: Append JSON instruction to system_prompt for Anthropic.

---

## API Summary

```python
def llm_inference(
    model: str,                           # Required: Model ID or name
    prompts: Union[str, list[str]],       # Required: Single or batch
    system_prompt: Optional[str] = None,  # Optional: System instruction
    max_tokens: int = 512,                # Default: 512
    temperature: float = 0.7,             # Default: 0.7 (balanced)
    top_p: float = 0.9,                   # Default: 0.9
    output_format: str = "text",          # "text" or "json"
    backend: str = "vllm",                # "vllm", "openai", "anthropic"
) -> dict[str, Any]:                      # Returns structured result
    """Run inference on any LLM (local via vLLM or remote via API)."""
```

**Output Schema**:
```python
{
    "outputs": ["generated text 1", "generated text 2", ...],
    "token_usage": {
        "input_tokens": 150,
        "output_tokens": 300,
    },
    "cost": 0.0045,  # USD (0.0 for vLLM)
}
```

---

## Implementation Roadmap

### Phase 1: Core Structure (Day 1)
- [ ] Create `llm_inference.py` file
- [ ] Implement task decorator and function signature
- [ ] Implement input validation
- [ ] Create helper function stubs

### Phase 2: Backend Implementation (Day 2-3)
- [ ] Implement vLLM backend (`_run_vllm`)
- [ ] Implement OpenAI backend (`_run_openai`)
- [ ] Implement Anthropic backend (`_run_anthropic`)
- [ ] Implement cost calculation functions

### Phase 3: Testing (Day 4)
- [ ] Write unit tests for validation
- [ ] Write integration tests with mocks
- [ ] Test each backend locally (requires API keys)
- [ ] Test batch processing and edge cases

### Phase 4: Docker & Deployment (Day 5)
- [ ] Update `genai-gpu/Dockerfile` to add Anthropic SDK
- [ ] Rebuild and push Docker image
- [ ] Create Kubernetes Secrets for API keys
- [ ] Register component in Flyte
- [ ] Verify end-to-end execution

### Phase 5: Documentation (Day 6)
- [ ] Update component README
- [ ] Add usage examples
- [ ] Update demo walkthrough if needed
- [ ] Close GitHub issue #96

---

## Dependencies & Prerequisites

### Python Packages (Already in genai-gpu image)
- ✅ `flytekit==1.14.3`
- ✅ `vllm==0.5.0`
- ✅ `openai==1.34.0`
- ⚠️ `anthropic==0.25.0` (needs to be added)

### Infrastructure
- ✅ EKS cluster deployed
- ✅ Flyte installed and configured
- ✅ GPU nodepools (for vLLM)
- ⚠️ vLLM server (needs deployment for testing)

### Environment Variables (for API backends)
- `OPENAI_API_KEY` (from Kubernetes Secret)
- `ANTHROPIC_API_KEY` (from Kubernetes Secret)

---

## Testing Strategy

### Unit Tests (No API Calls)
- Input validation for all parameters
- Edge cases (empty strings, invalid types)
- Cost calculation accuracy
- Prompt normalization (str → list)

### Integration Tests (Mocked APIs)
- vLLM response parsing
- OpenAI client interaction
- Anthropic client interaction
- Token counting aggregation
- Error handling for API failures

### End-to-End Tests (Real Execution)
- Local vLLM inference (requires GPU)
- OpenAI API inference (requires API key + costs money)
- Anthropic API inference (requires API key + costs money)
- Batch processing with 10+ prompts
- JSON mode validation

---

## Risk Assessment

### Low Risk ✅
- **Input validation**: Straightforward, well-tested patterns
- **vLLM integration**: Already using OpenAI client in `rag.py`
- **Cost calculation**: Simple arithmetic, testable

### Medium Risk ⚠️
- **API key management**: Requires proper Kubernetes Secret setup
- **Rate limiting**: OpenAI/Anthropic have rate limits (handle gracefully)
- **JSON mode on Anthropic**: Prompt engineering less reliable than native API

### Mitigation Strategies
- **API keys**: Document setup process, fail fast with clear errors
- **Rate limits**: Use `retries=2` in task decorator, add exponential backoff
- **JSON mode**: Document limitations, provide examples of effective prompts

---

## Success Metrics

### Quantitative
- ✅ All 3 backends functional (vLLM, OpenAI, Anthropic)
- ✅ Unit test coverage >80%
- ✅ Integration tests pass with mocked APIs
- ✅ End-to-end test succeeds with real vLLM deployment
- ✅ Cost estimation within 5% of actual API billing

### Qualitative
- ✅ API is intuitive and consistent with existing components
- ✅ Error messages are actionable and clear
- ✅ Documentation enables self-service implementation
- ✅ Component can be used in Act 3 demo (RAG pipeline)

---

## Next Steps

1. **Review Design**: Stakeholder review of this design package
2. **Address Feedback**: Incorporate any design changes
3. **Begin Implementation**: Follow Phase 1 of implementation roadmap
4. **Iterate**: Implement → Test → Refine cycle
5. **Deploy**: Push to production after E2E testing
6. **Document**: Update demo walkthrough with real code examples

---

## References

### Design Documents
- `llm_inference_design.md` - Complete design specification
- `IMPLEMENTATION_NOTES.md` - Implementation guide and code templates

### Related Components
- `embeddings.py` - Pattern reference for GenAI tasks
- `rag.py` - OpenAI client usage example
- `vllm_deploy.py` - vLLM deployment pattern

### External Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/claude/reference)
- [Flytekit Task Documentation](https://docs.flyte.org/projects/flytekit/en/latest/generated/flytekit.task.html)

---

## Contact & Questions

For questions about this design, contact:
- **Component Owner**: Jules Agent (implementing)
- **Technical Review**: ML Platform Team
- **Design Approval**: @xiaohanhuang

**Issue**: [#96 - COMPONENT: genai.llm_inference](https://github.com/xiaohanhuang/ml-platform/issues/96)

---

**Design Status**: ✅ Complete and Ready for Implementation

*Last Updated*: 2026-03-03 08:13 UTC
