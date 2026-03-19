# LLM Supervised Fine-Tuning (SFT) Workflow

End-to-end supervised fine-tuning pipeline for large language models with support for LoRA, QLoRA, and full fine-tuning.

## Features

- **Multiple Training Methods**: LoRA, QLoRA (4-bit/8-bit), and full parameter fine-tuning
- **HuggingFace Integration**: Load datasets and models from HuggingFace Hub
- **S3 Support**: Load custom datasets from S3
- **Production-Ready**: EFS checkpoints, MLflow registry, Teams notifications
- **GPU Optimized**: Automatic resource allocation based on model size

## Infrastructure Profiles

| Profile  | GPUs               | VRAM   | Target Models |
|:---------|:-------------------|:-------|:--------------|
| `small`  | 1x A10G            | 24 GB  | Up to 7B      |
| `medium` | 1x A100-80GB       | 80 GB  | Up to 13B     |
| `large`  | 4x A100-80GB (FSDP)| 320 GB | Up to 70B     |

## Pipeline Stages

1. **Data Loading**: Ingest from HuggingFace Hub or S3
2. **Tokenization**: Apply prompt templates and tokenize
3. **Data Splitting**: Split into train/val/test sets
4. **Training**: Fine-tune with LoRA, QLoRA, or full fine-tuning
5. **Evaluation**: Compute perplexity, ROUGE, accuracy
6. **Registry**: Register model in MLflow
7. **Notification**: Send Teams notification (optional)

## Quick Start

### 1. Build and Deploy

```bash
# Build Docker images
make build

# Push to ECR
make push

# Register with Flyte
make register

# Or do all at once
make deploy
```

### 2. Run a Workflow

**LoRA Fine-Tuning (7B model):**
```bash
ml-plat job submit \
  --workflow llm_sft_lora_pipeline \
  --base-model "meta-llama/Llama-3.1-8B" \
  --dataset "tatsu-lab/alpaca" \
  --epochs 3
```

**QLoRA Fine-Tuning (4-bit quantization):**
```bash
ml-plat job submit \
  --workflow llm_sft_lora_pipeline \
  --base-model "mistralai/Mistral-7B-v0.1" \
  --dataset "OpenAssistant/oasst1" \
  --quantization "4bit" \
  --epochs 5 \
  --learning-rate 1e-4
```

**Full Fine-Tuning:**
```bash
ml-plat job submit \
  --workflow llm_sft_full_pipeline \
  --base-model "meta-llama/Llama-2-70B" \
  --dataset "s3://my-bucket/custom-data.jsonl" \
  --epochs 1 \
  --batch-size 2
```

**With Teams Notification:**
```bash
ml-plat job submit \
  --workflow llm_sft_lora_pipeline \
  --base-model "meta-llama/Llama-3.1-8B" \
  --dataset "tatsu-lab/alpaca" \
  --epochs 3 \
  --teams-webhook "https://outlook.office.com/webhook/..."
```

## Parameters

### LoRA Pipeline (`llm_sft_lora_pipeline`)

| Parameter       | Type   | Default              | Description                                    |
|:----------------|:-------|:---------------------|:-----------------------------------------------|
| `base_model`    | str    | `Llama-3.1-8B`       | HuggingFace model ID                           |
| `dataset`       | str    | `tatsu-lab/alpaca`   | HuggingFace dataset or S3 URI                  |
| `epochs`        | int    | `3`                  | Number of training epochs                      |
| `learning_rate` | float  | `2e-4`               | Peak learning rate                             |
| `batch_size`    | int    | `4`                  | Per-device training batch size                 |
| `lora_r`        | int    | `16`                 | LoRA rank (LoRA/QLoRA only)                    |
| `lora_alpha`    | int    | `32`                 | LoRA alpha scaling (LoRA/QLoRA only)           |
| `teams_webhook` | str    | `""`                 | Optional Teams webhook URL for notifications   |

## Custom Datasets

### HuggingFace Format

Datasets should have one of these formats:

**Format 1: Alpaca-style**
```json
{"instruction": "...", "output": "..."}
```

**Format 2: Chat-style**
```json
{"system": "...", "user": "...", "assistant": "..."}
```

**Format 3: Raw text**
```json
{"text": "..."}
```

### S3 Format

Upload JSONL files to S3:
```bash
aws s3 cp my-dataset.jsonl s3://my-bucket/datasets/
```

Then reference in workflow:
```bash
--dataset "s3://my-bucket/datasets/my-dataset.jsonl"
```

## Monitoring

### MLflow

View experiments and models:
```bash
# Port-forward MLflow service
kubectl port-forward -n monitoring svc/mlflow 5000:5000

# Open browser
open http://localhost:5000
```

### Grafana

Monitor GPU utilization and training metrics:
```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80

# Open browser (admin/prom-operator)
open http://localhost:3000
```

## Checkpoints

Checkpoints are saved to EFS at `/mnt/efs/checkpoints/` with atomic writes for spot instance resilience.

To resume from a checkpoint:
```python
from transformers import AutoModelForCausalLM

# For LoRA adapters
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(base_model, "/mnt/efs/checkpoints/lora_finetune_final")

# For full models
model = AutoModelForCausalLM.from_pretrained("/mnt/efs/checkpoints/full_finetune_final")
```

## Cost Estimation

| Model Size | Profile  | Method | Duration | Est. Cost |
|:-----------|:---------|:-------|:---------|:----------|
| 1B         | small    | LoRA   | 20 min   | $0.50     |
| 7B         | small    | LoRA   | 90 min   | $2.25     |
| 13B        | medium   | LoRA   | 120 min  | $6.00     |
| 70B        | large    | Full   | 480 min  | $48.00    |

*Based on spot pricing for g5.xlarge and p4d.24xlarge instances.*

## Troubleshooting

### Out of Memory (OOM)

- Use QLoRA (`--method qlora`) for 4-bit quantization
- Reduce batch size (`--batch-size 2`)
- Increase gradient accumulation (`--gradient-accumulation-steps 8`)

### Slow Training

- Increase batch size if memory allows
- Use full fine-tuning with FSDP for multi-GPU parallelism
- Check GPU utilization in Grafana

### Dataset Errors

- Validate JSONL format (one valid JSON object per line)
- Ensure required fields exist (`instruction`, `output` or `text`)
- Check S3 permissions if using S3 datasets

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test with tiny model
make test-local
```

### Linting

```bash
ruff check .
mypy .
```

## Architecture

See [design document](../../../docs/recipes/llm-sft-design.md) for detailed architecture and component specifications.

## Dependencies

This workflow depends on the following components:

- `data.hf_dataset_loader` — HuggingFace dataset ingestion
- `data.tokenizer` — Tokenization with prompt templates
- `data.data_splitter` — Train/val/test splitting
- `training.lora_finetune` — LoRA fine-tuning
- `training.full_finetune` — Full parameter fine-tuning
- `evaluation.model_evaluator` — Model evaluation
- `model.registry_publisher` — MLflow registration
- `ops.notify_teams` — Teams notifications

## License

See repository root LICENSE file.
