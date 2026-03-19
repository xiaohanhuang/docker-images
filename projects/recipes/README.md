# RLHF Alignment Recipe — README

**Recipe Name**: `llm-rlhf`
**Status**: 📋 Design Complete → 🚧 Implementation Pending
**GitHub Issue**: [#113](https://github.com/xiaohanhuang/ml-platform/issues/113)

---

## Overview

The `llm-rlhf` recipe provides a complete pipeline for aligning large language models using Reinforcement Learning from Human Feedback (RLHF). This is the same technique used to create models like ChatGPT, Claude, and Llama-Chat.

**Pipeline Steps**:
1. **Data Ingestion**: Load instruction dataset from HuggingFace
2. **Supervised Fine-Tuning (SFT)**: Fine-tune base model on instructions
3. **Preference Generation**: Generate preference pairs using LLM judge
4. **Reward Model Training**: Train model to score outputs
5. **Reward Evaluation**: Validate reward model quality
6. **PPO Alignment**: Optimize policy using reward feedback
7. **Model Evaluation**: Evaluate aligned model
8. **Registry Publishing**: Register final model

**Total Runtime**: ~2 hours (small profile) | ~4 hours (medium) | ~8 hours (large)

---

## Quick Start

> **Note**: The recipe engine can parse and validate this recipe definition, but full
> end-to-end execution is **not yet supported**. Several pipeline components
> (`training.finetune_lm`, `data.preference_data_generator`, `training.reward_model`,
> `training.ppo_trainer`, etc.) are placeholders and have not been implemented.
> The `data.hf_dataset_loader` component is functional. Use `ml-plat recipe validate`
> to check the recipe definition, and track progress in
> [issue #113](https://github.com/xiaohanhuang/ml-platform/issues/113).

### Prerequisites

- ML Platform deployed with:
  - Flyte (workflow orchestration)
  - Ray (distributed training)
  - Karpenter (GPU autoscaling)
  - MLflow (model registry)

- API keys (if using external judge):
  - OpenAI API key (`OPENAI_API_KEY`)
  - Anthropic API key (`ANTHROPIC_API_KEY`)

### Run the Recipe

> **Note**: `ml-plat recipe run` generates Flyte workflow code and saves it
> locally. End-to-end Flyte execution is not yet implemented — use
> `--dry-run` to preview the generated workflow, then register and run it
> manually with `pyflyte`.

```bash
# Dry-run: preview generated workflow code without submitting
ml-plat recipe run llm-rlhf --dry-run \
    --base-model meta-llama/Llama-3-8b \
    --dataset Anthropic/hh-rlhf \
    --model-name llama3-8b-helpful-v1

# Basic usage (small 7B model, 2 GPUs)
ml-plat recipe run llm-rlhf \
    --base-model meta-llama/Llama-3-8b \
    --dataset Anthropic/hh-rlhf \
    --model-name llama3-8b-helpful-v1

# Medium profile (13B model, 4 GPUs)
ml-plat recipe run llm-rlhf \
    --base-model meta-llama/Llama-3-13b \
    --dataset Anthropic/hh-rlhf \
    --profile medium \
    --model-name llama3-13b-helpful-v1

# Large profile (70B model, 16 GPUs with EFA)
ml-plat recipe run llm-rlhf \
    --base-model meta-llama/Llama-3-70b \
    --dataset Anthropic/hh-rlhf \
    --profile large \
    --model-name llama3-70b-helpful-v1
```

---

## Configuration

### Infrastructure Profiles

| Profile | Model Size | Total GPUs | Instance Type | Estimated Cost/Hour |
|:---|:---|:---:|:---|:---:|
| `small` | 7B | 2 | g5.xlarge | $2.50 |
| `medium` | 13B | 4 | p4d.24xlarge | $10.00 |
| `large` | 70B | 16 | p4d.24xlarge | $48.00 |

### Common Parameters

```bash
# Dataset configuration
--dataset Anthropic/hh-rlhf           # HuggingFace dataset name
--dataset-subset helpful-base         # Optional subset
--num-train-samples 10000             # Number of samples to use

# Training hyperparameters
--sft-epochs 3                        # SFT epochs (default: 3)
--sft-lr 2e-4                         # SFT learning rate
--lora-rank 16                        # LoRA rank (4-128)

--rm-epochs 1                         # Reward model epochs (default: 1)
--rm-lr 1e-5                          # Reward model learning rate

--ppo-epochs 1                        # PPO outer epochs (default: 1)
--ppo-rollouts 1024                   # Rollouts per epoch
--kl-coef 0.1                         # KL divergence penalty

# Judge configuration
--judge-model gpt-4                   # LLM judge (gpt-4, claude-3-opus, etc.)
--judge-criteria helpfulness          # Evaluation criteria

# Model registry
--registry-type mlflow                # Registry backend (mlflow/s3/huggingface)
```

### Complete Example

```bash
ml-plat recipe run llm-rlhf \
    --base-model meta-llama/Llama-3-8b \
    --dataset Anthropic/hh-rlhf \
    --dataset-subset helpful-base \
    --num-train-samples 5000 \
    --profile small \
    --sft-epochs 3 \
    --sft-lr 2e-4 \
    --lora-rank 16 \
    --rm-epochs 1 \
    --rm-lr 1e-5 \
    --ppo-epochs 1 \
    --ppo-rollouts 1024 \
    --kl-coef 0.1 \
    --judge-model gpt-4 \
    --judge-criteria helpfulness \
    --model-name llama3-8b-helpful-base-v1 \
    --registry-type mlflow
```

---

## Datasets

### Recommended Datasets

**Helpfulness**:
- `Anthropic/hh-rlhf` (subset: `helpful-base`, `helpful-rejection-sampled`)
- `openai/summarize_from_feedback`
- `stanfordnlp/shp`

**Harmlessness**:
- `Anthropic/hh-rlhf` (subset: `harmless-base`)
- `PKU-Alignment/PKU-SafeRLHF`

**General Instruction Following**:
- `OpenAssistant/oasst1`
- `tatsu-lab/alpaca`
- `databricks/databricks-dolly-15k`

### Dataset Format

Input dataset must be in JSONL format with one of the following structures:

**Format 1: Prompt field**
```json
{"prompt": "Write a poem about AI", "text": "optional completion"}
```

**Format 2: Messages field (ChatML)**
```json
{"messages": [
  {"role": "user", "content": "Write a poem about AI"},
  {"role": "assistant", "content": "In circuits deep..."}
]}
```

The `hf_dataset_loader` component automatically converts HuggingFace datasets to this format.

---

## Output

### Final Model Location

After successful completion, the aligned model is registered in MLflow:

```bash
# Get model URI from command output
Model URI: models:/llama3-8b-helpful-v1/1

# Load model in Python
import mlflow
model_uri = "models:/llama3-8b-helpful-v1/1"
model = mlflow.pyfunc.load_model(model_uri)
```

### Metrics Tracked

**SFT Metrics** (Step 2):
- Training loss
- Epochs
- Learning rate

**Reward Model Metrics** (Steps 4-5):
- Validation accuracy (% of correct preference rankings)
- Mean margin (average reward gap between chosen/rejected)
- Final loss

**PPO Metrics** (Step 6):
- Mean reward (per epoch)
- KL divergence (vs. reference model)
- Policy loss
- Value loss

**Alignment Metrics** (Step 7):
- Helpfulness score (1-10)
- Harmlessness score (1-10)
- Honesty score (1-10)

### Visualize in Flyte UI

```bash
# Get execution URL from command output
Flyte Execution: https://flyte.example.com/console/projects/.../executions/...

# Or list recent executions
ml-plat flyte list-executions --project ml-platform --limit 5
```

---

## Monitoring

### Resource Usage

Monitor GPU utilization in Grafana:
```bash
# Open Grafana dashboard
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80

# Navigate to: http://localhost:3000/d/nvidia-dcgm/
# Credentials: admin / (get from secret)
```

### Cost Tracking

Track estimated costs:
```bash
# View cost for last execution
ml-plat cost --last-execution

# View cost for specific time window
ml-plat cost --start "2026-03-03T00:00:00" --end "2026-03-03T23:59:59"
```

### Ray Dashboard

Monitor distributed training:
```bash
# Get Ray head pod
kubectl get pods -n flyte | grep ray-head

# Port forward to Ray dashboard
kubectl port-forward -n flyte <ray-head-pod> 8265:8265

# Navigate to: http://localhost:8265
```

---

## Troubleshooting

### Common Issues

**Issue**: Recipe validation fails
```bash
Error: Missing required parameter 'base_model'
```
**Solution**: Ensure all required parameters are provided:
```bash
--base-model <model>
--dataset <dataset>
--model-name <name>
```

---

**Issue**: GPU nodes not provisioning
```bash
Error: Pods pending for >10 minutes
```
**Solution**: Check Karpenter node pools:
```bash
kubectl get nodepools
kubectl describe nodepool gpu-nodepool
```

Ensure GPU nodepool has capacity and correct taints/labels.

---

**Issue**: Reward model accuracy too low (<60%)
```bash
Warning: Reward model validation accuracy: 52.3%
```
**Solution**:
1. Increase training epochs: `--rm-epochs 3`
2. Use more preference data: `--num-train-samples 20000`
3. Try different base model size (larger = better reward model)

---

**Issue**: KL divergence too high during PPO (>5.0)
```bash
Warning: KL divergence: 7.2 (threshold: 5.0)
```
**Solution**:
1. Increase KL penalty: `--kl-coef 0.2`
2. Decrease learning rate: `--ppo-lr 5e-6`
3. Reduce PPO inner epochs: `--ppo-inner-epochs 2`

---

**Issue**: Out of memory during PPO training
```bash
Error: CUDA out of memory
```
**Solution**:
1. Use larger profile: `--profile medium`
2. Enable LoRA for reward/critic models (reduce memory)
3. Reduce batch size in component configuration

---

### Debug Mode

Run with verbose logging:
```bash
ml-plat recipe run llm-rlhf \
    --base-model <model> \
    --dataset <dataset> \
    --model-name <name> \
    --verbose \
    --log-level DEBUG
```

---

## Advanced Usage

### Custom Judge Prompts

Override default judge prompt by modifying the preference generator component:

```python
# Edit: projects/components/components/data/preference_generator.py
JUDGE_PROMPT_TEMPLATE = """
Your custom judge prompt here.

Prompt: {prompt}
Response A: {response_a}
Response B: {response_b}

Which is better?
"""
```

---

### Resume from Checkpoint

If execution fails mid-pipeline, resume from the last successful step:

```bash
ml-plat recipe resume <execution-id> --from-step train_reward_model
```

*(Note: Resume functionality requires Recipe Engine implementation)*

---

### Multi-Objective Alignment

Optimize for multiple objectives simultaneously:

```bash
ml-plat recipe run llm-rlhf \
    --base-model <model> \
    --dataset Anthropic/hh-rlhf \
    --model-name llama3-8b-helpful-harmless-v1 \
    --judge-criteria "helpfulness,harmlessness" \
    --multi-objective-weights "0.7,0.3"
```

*(Note: Multi-objective support requires RLHF Trainer enhancement)*

---

## Performance Optimization

### Reduce Training Time

1. **Use fewer samples during development**:
   ```bash
   --num-train-samples 1000  # vs. default 10000
   ```

2. **Reduce epochs**:
   ```bash
   --sft-epochs 1 --rm-epochs 1 --ppo-epochs 1
   ```

3. **Enable Flyte caching**:
   ```bash
   # Components are cached by default
   # Re-running with same inputs reuses previous outputs
   ```

4. **Use warm GPU pools**:
   - Ensure warm pool is enabled (see `projects/eks/warm-pool.yaml`)
   - Reduces node provisioning time from ~2 minutes to ~4 seconds

---

### Reduce Cost

1. **Use spot instances** (when supported):
   ```bash
   # Modify Karpenter nodepool to use spot
   # Edit: projects/eks/karpenter-nodepool.yaml
   # Set: karpenter.sh/capacity-type: spot
   ```

2. **Use smaller models**:
   ```bash
   --base-model meta-llama/Llama-3-8b  # vs. Llama-3-70b
   ```

3. **Scale down when idle**:
   - KEDA automatically scales to zero when no work
   - Warm pool scales down during off-hours (see CronJob schedules)

---

## Example Workflows

### Minimal Test Run (Fast)

For testing the pipeline end-to-end:

```bash
ml-plat recipe run llm-rlhf \
    --base-model distilgpt2 \
    --dataset openai/summarize_from_feedback \
    --num-train-samples 100 \
    --sft-epochs 1 \
    --rm-epochs 1 \
    --ppo-epochs 1 \
    --ppo-rollouts 128 \
    --model-name distilgpt2-aligned-test
```

**Expected Runtime**: ~30 minutes
**Expected Cost**: ~$1

---

### Production Run (High Quality)

For production-grade aligned model:

```bash
ml-plat recipe run llm-rlhf \
    --base-model meta-llama/Llama-3-8b \
    --dataset Anthropic/hh-rlhf \
    --dataset-subset helpful-base \
    --num-train-samples 50000 \
    --profile small \
    --sft-epochs 3 \
    --rm-epochs 2 \
    --ppo-epochs 3 \
    --ppo-rollouts 2048 \
    --kl-coef 0.05 \
    --judge-model gpt-4 \
    --model-name llama3-8b-helpful-production-v1
```

**Expected Runtime**: ~6 hours
**Expected Cost**: ~$15

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────┐
│                  Recipe Engine                  │
│  (Parses YAML → Generates Flyte Workflow)       │
└────────────────────┬────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────┐
│              Flyte Orchestration                │
│  (Executes DAG, manages dependencies)           │
└─────┬──────┬──────┬──────┬──────┬──────┬───────┘
      │      │      │      │      │      │
      v      v      v      v      v      v
   ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
   │ 1  │→│ 2  │→│ 3  │→│ 4  │→│ 5  │→│ 6  │→...
   │Data│ │SFT │ │Pref│ │RM  │ │Eval│ │PPO │
   └────┘ └────┘ └────┘ └────┘ └────┘ └────┘
     ↓      ↓      ↓      ↓      ↓      ↓
   ┌────────────────────────────────────────┐
   │     Kubernetes (EKS) + Karpenter       │
   │  (Provisions GPUs, schedules pods)      │
   └────────────────────────────────────────┘
```

### GPU Topology (PPO Step)

```
┌──────────────────────────────────────────┐
│         Ray Cluster (PPO Training)       │
├──────────────────────────────────────────┤
│                                          │
│  GPU 0: Actor Model (trainable)         │
│         Reference Model (frozen)         │
│                                          │
│  GPU 1: Critic Model (trainable)        │
│         Reward Model (frozen)            │
│                                          │
└──────────────────────────────────────────┘

* Small profile (2 GPUs): Models share GPUs
* Medium profile (4 GPUs): Each model gets dedicated GPU
* Large profile (16 GPUs): Models sharded across multiple GPUs
```

---

## Related Documentation

- **Design Document**: `docs/rlhf-pipeline-design.md`
- **Implementation Guide**: `docs/rlhf-implementation-guide.md`
- **Recipe Definition**: `projects/recipes/llm-rlhf.yaml`
- **Demo Script**: `demo_walkthrough.md` (Act 5)
- **Component Patterns**: `projects/components/README.md`

---

## Contributing

### Adding New Components

See `docs/rlhf-implementation-guide.md` for component implementation checklist.

**Component Template**:
```python
"""
Component description.

Image: genai-gpu (or ml-gpu, data-cpu, etc.)
"""

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

@task(
    retries=1,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(cpu="8", mem="32Gi", gpu="1"),
    cache=True,
    cache_version="1.0",
)
def my_component(input_param: FlyteFile) -> FlyteFile:
    """Component docstring with Args and Returns.

    Heavy imports inside task body to avoid loading at import time.
    """
    import torch  # Lazy import

    # Implementation
    result_path = "/tmp/output.jsonl"
    # ... write results ...

    return FlyteFile(path=result_path)
```

### Testing Components

```bash
# Run component unit tests
pytest tests/components/test_my_component.py -v

# Run integration tests (requires GPU)
pytest tests/integration/test_rlhf_pipeline.py -v -m gpu

# Run recipe validation
ml-plat recipe validate llm-rlhf --params config.json
```

---

## Support

**Issues**: https://github.com/xiaohanhuang/ml-platform/issues/113
**Documentation**: `docs/rlhf-*.md`
**Demo**: `demo_walkthrough.md` (Act 5)

---

**Version**: 1.0.0
**Last Updated**: 2026-03-03
**Status**: 📋 Design Complete → 🚧 Implementation Pending
