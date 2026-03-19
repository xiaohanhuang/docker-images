# Preference Data Generator

AI-powered preference pair creation for DPO/RLHF training.

## Overview

The `generate_preference_data` component automates the most expensive and time-consuming part of alignment training: creating preference datasets. Instead of requiring human annotators to manually rank model responses, this component:

1. Generates multiple candidate responses from your target model
2. Uses a judge LLM to rank responses based on customizable criteria
3. Creates preference pairs (chosen/rejected) for DPO training or full rankings for more advanced RLHF methods

## Why This Matters

Creating preference data is typically the bottleneck in RLHF/DPO workflows:
- **Manual annotation is expensive**: Human labelers cost $20-50/hour
- **Annotation is time-consuming**: Even simple preference judgments take 30-60 seconds per example
- **Quality varies**: Inter-annotator agreement is often low (60-70% agreement is common)

This component uses AI judges to create preference data at scale, making RLHF/DPO accessible without human annotators.

## Component Details

- **Category**: Data / RL
- **Image**: `genai-gpu`
- **Location**: `projects/components/components/data/preference_data_generator.py`

## Inputs

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `prompt_data_path` | str | *required* | S3 path to prompt dataset (JSONL format) |
| `s3_output_path` | str | *required* | S3 path where preference dataset will be saved |
| `generator_model` | str | *required* | Model to generate candidate responses |
| `prompt_column` | str | `"prompt"` | Column name with prompts in input JSONL |
| `judge_model` | str | `"gpt-4o"` | Judge model for ranking responses |
| `n_candidates` | int | `4` | Number of responses to generate per prompt |
| `judge_criteria` | str | `"helpfulness, accuracy, safety"` | Ranking criteria |
| `output_format` | str | `"dpo"` | Output format: "dpo" or "ranking" |

### Model Specifications

**Generator Model** (`generator_model`):
- HuggingFace model ID (e.g., `"meta-llama/Llama-3-8b-instruct"`)
- With `hf://` prefix (e.g., `"hf://gpt2"`)
- vLLM endpoint: `"vllm://model-name@http://endpoint:8000"`

**Judge Model** (`judge_model`):
- OpenAI models: `"gpt-4o"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"`
- vLLM endpoint: `"vllm://model-name@http://endpoint:8000"`
- HuggingFace model: `"hf://model-id"` (uses heuristic ranking)
- Simple heuristic: `"heuristic"` (no external model needed)

**Note**: When using OpenAI judge models, set the `OPENAI_API_KEY` environment variable.

## Outputs

| Name | Type | Description |
|:--|:--|:--|
| `s3_path` | str | S3 path to preference dataset |
| `num_pairs` | int | Number of preference pairs created |
| `avg_score_delta` | float | Average score difference between chosen/rejected |

## Output Format

### DPO Format (`output_format="dpo"`)

Each line in the output JSONL contains a preference pair:

```json
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum bits (qubits) that can exist in superposition...",
  "rejected": "Quantum computers are faster computers.",
  "chosen_score": 8.5,
  "rejected_score": 3.2
}
```

This format is compatible with:
- TRL's `DPOTrainer`
- Hugging Face `trl` library
- Standard RLHF pipelines

### Ranking Format (`output_format="ranking"`)

Each line contains a full ranking of all candidates:

```json
{
  "prompt": "Explain quantum computing",
  "candidates": [
    {
      "response": "Quantum computing uses quantum bits...",
      "score": 8.5,
      "rank": 1
    },
    {
      "response": "Quantum computers are special machines...",
      "score": 6.8,
      "rank": 2
    },
    {
      "response": "Quantum computers are faster computers.",
      "score": 3.2,
      "rank": 3
    }
  ]
}
```

This format is useful for:
- Advanced RLHF methods (PPO, ReMax)
- Training reward models with ranking loss
- Analysis and debugging

## Usage Example

### Simple DPO Dataset Creation

```python
from flytekit import workflow
from data.preference_data_generator import generate_preference_data

@workflow
def create_dpo_dataset_wf(prompts_s3: str, output_s3: str) -> str:
    """Create DPO preference dataset from prompts."""
    s3_path, num_pairs, score_delta = generate_preference_data(
        prompt_data_path=prompts_s3,
        s3_output_path=output_s3,
        generator_model="meta-llama/Llama-3-8b-instruct",
        judge_model="gpt-4o",
        n_candidates=4,
        judge_criteria="helpfulness, accuracy, safety",
        output_format="dpo",
    )
    return s3_path
```

### Using vLLM Endpoints

If you have a vLLM deployment:

```python
s3_path, num_pairs, score_delta = generate_preference_data(
    prompt_data_path="s3://my-bucket/prompts.jsonl",
    s3_output_path="s3://my-bucket/preferences.jsonl",
    generator_model="vllm://llama-3-8b@http://vllm-service:8000",
    judge_model="vllm://judge-model@http://judge-service:8000",
    n_candidates=4,
)
```

### Using Heuristic Judge (No External API)

For quick iteration without external judge costs:

```python
s3_path, num_pairs, score_delta = generate_preference_data(
    prompt_data_path="s3://my-bucket/prompts.jsonl",
    s3_output_path="s3://my-bucket/preferences.jsonl",
    generator_model="meta-llama/Llama-3-8b-instruct",
    judge_model="heuristic",  # No external API needed
    n_candidates=4,
)
```

## Input Data Format

Your prompt dataset should be JSONL with at least a prompt column:

```json
{"prompt": "Explain quantum computing"}
{"prompt": "What is machine learning?"}
{"prompt": "How does photosynthesis work?"}
```

Or with a custom column name:

```json
{"question": "Explain quantum computing", "context": "For a high school student"}
{"question": "What is machine learning?", "context": "Technical audience"}
```

Then specify `prompt_column="question"` when calling the component.

## Resource Requirements

- **CPU**: 8 cores (request), 16 cores (limit)
- **Memory**: 32 GiB (request), 64 GiB (limit)
- **GPU**: 1 GPU (for generator model)
- **Image**: `genai-gpu` (includes transformers, vLLM, OpenAI client)

## Cost Optimization

### Using Heuristic Judge
- **Pros**: No API costs, fast iteration
- **Cons**: Lower quality rankings than LLM judges
- **Best for**: Initial prototyping, testing pipelines

### Using GPT-4 Judge
- **Pros**: High-quality rankings, good agreement with humans
- **Cons**: $0.01-0.03 per prompt (with 4 candidates)
- **Best for**: Production datasets, final training data

### Using vLLM Judge
- **Pros**: No per-request costs after deployment, fast
- **Cons**: Requires deploying and maintaining judge model
- **Best for**: Large-scale generation (>10k examples)

## Tips & Best Practices

1. **Start small**: Generate 100-200 examples first to validate quality
2. **Tune judge criteria**: Adjust `judge_criteria` based on your use case
3. **Validate outputs**: Sample and manually review 50-100 pairs
4. **Monitor score deltas**: Low `avg_score_delta` (<2.0) suggests weak differentiation
5. **Use multiple judges**: Run with different judges and compare agreement
6. **Cache results**: Enable Flyte caching to avoid regenerating data

## Common Issues

### Low Score Differentiation
If `avg_score_delta` is very low (<1.0):
- The generator might be producing similar responses
- Increase `n_candidates` to get more variety
- Try different generation parameters (temperature, top_p)

### High Judge Variance
If different judges produce very different rankings:
- Your criteria might be ambiguous
- Clarify `judge_criteria` with more specific instructions
- Use a stronger judge model

### Out of Memory (OOM)
If the generator runs out of GPU memory:
- Use a smaller model
- Reduce batch size in generator inference
- Use quantization (8-bit or 4-bit)

## Testing

Run tests with:

```bash
pytest tests/components/test_preference_data_generator.py -v
```

Tests cover:
- S3 URI validation
- Model specification parsing
- Judge ranking logic
- DPO and ranking output formats
- Edge cases (empty files, invalid formats)

## Next Steps

After generating preference data:

1. **Train with DPO**: Use `trl.DPOTrainer` or similar
2. **Train reward model**: Use ranking format with ranking loss
3. **RLHF pipeline**: Feed into PPO or other RL algorithms

## Related Components

- `training.finetune`: Fine-tune models before generating preferences
- `evaluation.eval_lm`: Evaluate models after DPO training
- `genai.rag`: Use similar judge patterns for RAG evaluation
