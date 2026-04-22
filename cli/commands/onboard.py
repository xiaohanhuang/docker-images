"""
Interactive onboarding wizard for new ML engineers.

Guides users from zero to running a GPU training job in minutes,
with templates that use real platform SDK patterns, correct images,
and an AI assistant grounded in actual platform knowledge.
"""

import os
import re
import textwrap
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()

# ── ECR constants ─────────────────────────────────────────────────────────
# NOTE: ECR_REGISTRY is read from environment variable or config at runtime.
# This default is for template generation only and should match your AWS account.
ECR_REGISTRY = os.getenv("ECR_REGISTRY", "805673386114.dkr.ecr.us-west-2.amazonaws.com")
ECR_REPO = "ml-platform"

# ═══════════════════════════════════════════════════════════════════════════
# Platform knowledge base for the AI assistant
# ═══════════════════════════════════════════════════════════════════════════

PLATFORM_SYSTEM_PROMPT = textwrap.dedent("""\
You are the ML Platform onboarding assistant. You have deep knowledge of this
specific production ML platform built on AWS EKS.

== Architecture ==
The platform combines Flyte (workflow orchestration), Ray (distributed training),
Spark (ETL), and Kubernetes (EKS) with full observability via Prometheus, Grafana,
MLflow, and dcgm-exporter for GPU metrics.

== Docker Image Hierarchy (3-layer) ==
Layer 1 (base):
  - base-cpu  (python:3.12-slim + system tools)
  - base-gpu  (NVIDIA CUDA 12.9.1 + PyTorch 2.10.0)
Layer 2 (framework):
  - flyte-cpu (base-cpu + flytekit 1.16.14 + boto3 + s3fs)
  - flyte-gpu (base-gpu + flytekit 1.16.14 + boto3 + s3fs)
  - ray-worker (base-gpu + ray 2.54.0 + pandas + pyarrow)
Layer 3 (workload):
  - data-cpu   (flyte-cpu + pandas + pyarrow + scikit-learn + tiktoken + bs4)
  - ml-gpu     (flyte-gpu + transformers 5.3 + accelerate + peft + datasets + mlflow 3.10)
  - genai-gpu  (flyte-gpu + vllm 0.16 + langchain 1.2 + sentence-transformers 5.2 + chromadb 1.5)
  - training-llm (ray-worker + transformers + peft + accelerate + bitsandbytes + deepspeed)
All images live in ECR: 805673386114.dkr.ecr.us-west-2.amazonaws.com/ml-platform/<name>:latest

== SDK (ml_platform_sdk) ==
- ml_platform_sdk.tasks.training: @task with RayJobConfig for distributed GPU training
- ml_platform_sdk.tasks.efs: @efs_task decorator auto-mounts EFS PVC at /mnt/efs
- ml_platform_sdk.tasks.data: download_dataset (S3 → FlyteFile)
- ml_platform_sdk.tasks.spark: spark_task decorator
- ml_platform_sdk.profiling: profile() context manager for torch.profiler
- ml_platform_sdk.remote: @remote decorator for zero-config GPU execution without Docker

== Reusable Components ==
- components/data/text_chunker.py → chunk_documents(s3_input, s3_output, strategy, chunk_size)
- components/genai/vector_store_indexer.py → index_embeddings(
    embeddings_path, backend="pgvector"|"faiss"|"chromadb")
- components/training/, components/evaluation/, components/serving/

== CLI Commands (mlp) ==
- mlp init              — initial config (~/.mlp/config.yaml)
- mlp job submit        — submit Flyte workflow
- mlp job status        — check job status
- mlp job logs          — stream logs
- mlp pod launch        — launch GPU pod for interactive work
- mlp pod connect       — SSH into running pod
- mlp notebook open     — open JupyterHub
- mlp debug start       — start GPU debug pod
- mlp cost estimate     — estimate job cost
- mlp wizard            — this wizard

== Code Conventions ==
1. Lazy imports: Heavy deps (torch, mlflow, boto3) imported INSIDE task bodies
2. Caching: @task(cache=True, cache_version="X") for deterministic tasks
3. EFS: Mount via @efs_task or ml_platform_sdk.tasks.efs for checkpoints/datasets
4. GPU tolerations: nvidia.com/gpu=true:NoSchedule, nodeSelector: role: gpu-worker
5. Atomic checkpoints: save to .tmp → os.fsync() → os.rename() for spot resilience
6. Container images: Use the 3-layer hierarchy, pick the right workload image

== Example Workflows ==
01_quickstart:  prepare_data → train_distributed (Ray 2 GPU) → evaluate_model
02_llm_finetune: validate → finetune (Ray 4 GPU, training-llm image) → evaluate
03_spark_etl:   Spark extract → Spark feature_eng → Ray train
04_notebook:    Interactive JupyterHub exploration
05_ray_distributed: CIFAR-10 ResNet-18, 2 GPU, pod anti-affinity, MLflow
06_pytorch_ddp: Native PyTorch DDP via KFPyTorch, EFS checkpoints
07_remote_exec: @remote(gpu=1) zero-config GPU execution

== Storage ==
- S3: datasets, artifacts, model checkpoints
- EFS PVC (efs-claim) at /mnt/efs: shared durable storage for checkpoints
- ECR: container images

Answer concisely with platform-specific details. Reference the correct images,
SDK functions, CLI commands, and code patterns. Give working code examples
whenever possible.
""")


# ═══════════════════════════════════════════════════════════════════════════
# AWS Bedrock LLM Client
# ═══════════════════════════════════════════════════════════════════════════


class BedrockAssistant:
    """LLM assistant using AWS Bedrock, grounded in platform knowledge."""

    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self.model_id = "deepseek.v3.2"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def ask(self, question: str, context: str = "") -> str:
        user_message = question
        if context:
            user_message = f"Context: {context}\n\nQuestion: {question}"

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                system=[{"text": PLATFORM_SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": 1500,
                    "temperature": 0.3,
                    "topP": 0.9,
                },
            )
            return response["output"]["message"]["content"][0]["text"].strip()

        except Exception as e:
            console.print(
                f"[yellow]⚠️  LLM assistant unavailable: {e}[/yellow]",
                style="dim",
            )
            return (
                "I'm unable to answer right now. "
                "Check docs/ or examples/ in the repo, or ask your team."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Project Template Generators
# ═══════════════════════════════════════════════════════════════════════════


def _write_files(project_dir: Path, files: dict[str, str]):
    """Write a dict of {relative_path: content} into project_dir."""
    project_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        p = project_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    # Automatically ignore generated project directories in the nearest .gitignore
    gitignore = project_dir.parent / ".gitignore"
    if gitignore.exists():
        entry = f"{project_dir.name}/"
        existing = gitignore.read_text()
        if entry not in existing.splitlines():
            with gitignore.open("a") as f:
                f.write(f"\n# mlp generated project\n{entry}\n")


def generate_llm_finetune_project(project_name: str, base_dir: Path) -> Path:
    """Generate a fine-tuning project using the training-llm image + Ray."""
    project_dir = base_dir / project_name

    workflow_py = textwrap.dedent('''\
        """
        LLM Fine-tuning Pipeline

        DAG: [Validate Dataset] → [Fine-tune with LoRA on Ray] → [Evaluate Perplexity]

        Image: training-llm (ray-worker + transformers + peft + deepspeed)
        """

        from flytekit import Resources, task, workflow
        from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

        TRAINING_IMAGE = (
            "{{ECR_REGISTRY}}/{{ECR_REPO}}/training-llm:1.1.0"
        )

        llm_ray_config = RayJobConfig(
            head_node_config=HeadNodeConfig(
                ray_start_params={"dashboard-host": "0.0.0.0"},
                requests=Resources(cpu="4", mem="16Gi"),
            ),
            worker_node_config=[
                WorkerNodeConfig(
                    group_name="gpu-workers",
                    replicas=2,
                    min_replicas=1,
                    max_replicas=4,
                    requests=Resources(cpu="8", mem="64Gi", gpu="1"),
                )
            ],
        )


        @task(
            requests=Resources(cpu="2", mem="8Gi"),
            cache=True,
            cache_version="1.0",
        )
        def validate_dataset(s3_path: str) -> str:
            """Validate dataset format and return path if valid."""
            # Lazy import: boto3 is heavy, only load inside the task body
            import boto3

            s3 = boto3.client("s3")
            # Parse s3://bucket/key
            parts = s3_path.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
            resp = s3.head_object(Bucket=bucket, Key=key)
            size_mb = resp["ContentLength"] / (1024 * 1024)
            print(f"✓ Dataset exists: {s3_path} ({size_mb:.1f} MB)")
            return s3_path


        @task(
            task_config=llm_ray_config,
            requests=Resources(cpu="4", mem="16Gi"),
            container_image=TRAINING_IMAGE,
        )
        def finetune_task(
            dataset_path: str,
            model_name: str,
            epochs: int,
            batch_size: int,
        ) -> str:
            """Fine-tune LLM with LoRA using Ray TorchTrainer on GPU workers."""
            # Lazy imports — only loaded inside the task body at runtime
            import ray.train
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer

            output_dir = "s3://{{S3_BUCKET}}/models/{{PROJECT_NAME}}"

            def train_loop(config):
                """Runs on each Ray GPU worker."""
                import torch
                from datasets import load_dataset
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    Trainer,
                    TrainingArguments,
                )

                rank = ray.train.get_context().get_world_rank()

                tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
                tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    config["model_name"],
                    torch_dtype=torch.bfloat16,
                    device_map={"" : ray.train.get_context().get_local_rank()},
                )
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                )
                model = get_peft_model(model, lora_config)
                if rank == 0:
                    model.print_trainable_parameters()

                dataset = load_dataset("json", data_files=config["dataset_path"], split="train")

                def tokenize(example):
                    return tokenizer(
                        example["text"],
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                    )

                tokenized = dataset.map(
                    tokenize, batched=True, remove_columns=dataset.column_names
                )

                training_args = TrainingArguments(
                    output_dir="/tmp/checkpoints",
                    num_train_epochs=config["epochs"],
                    per_device_train_batch_size=config["batch_size"],
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    bf16=True,
                    logging_steps=10,
                    save_strategy="epoch",
                    report_to="mlflow" if rank == 0 else "none",
                )
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized,
                    tokenizer=tokenizer,
                )
                trainer.train()
                if rank == 0:
                    trainer.save_model(config["output_dir"])
                ray.train.report({"status": "done"})

            trainer = TorchTrainer(
                train_loop_per_worker=train_loop,
                train_loop_config={
                    "model_name": model_name,
                    "dataset_path": dataset_path,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "output_dir": output_dir,
                },
                scaling_config=ScalingConfig(
                    num_workers=2,
                    use_gpu=True,
                    resources_per_worker={"GPU": 1},
                ),
            )
            trainer.fit()
            return output_dir


        @task(requests=Resources(cpu="4", mem="16Gi", gpu="1"))
        def evaluate_perplexity(model_path: str) -> float:
            """Calculate perplexity on a held-out test set.

            Expects an eval.jsonl file in the model checkpoint directory
            (e.g. s3://bucket/models/project/eval.jsonl) with one JSON object
            per line containing a "text" field.
            """
            import math

            import mlflow
            import torch
            from datasets import load_dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model.eval()

            dataset = load_dataset("json", data_files=f"{model_path}/eval.jsonl", split="train")
            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for sample in dataset.select(range(min(100, len(dataset)))):
                    inputs = tokenizer(
                        sample["text"], return_tensors="pt", truncation=True, max_length=512
                    ).to(model.device)
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item() * inputs["input_ids"].numel()
                    total_tokens += inputs["input_ids"].numel()

            perplexity = math.exp(total_loss / max(total_tokens, 1))
            mlflow.log_metric("eval_perplexity", perplexity)
            print(f"Perplexity: {perplexity:.4f}")
            return perplexity


        @workflow
        def llm_finetune_pipeline(
            dataset: str = "s3://{{S3_BUCKET}}/datasets/train.jsonl",
            model_name: str = "{{MODEL_NAME}}",
            epochs: int = 3,
            batch_size: int = 4,
        ) -> float:
            """Production LLM fine-tuning pipeline.

            Submit:
                mlp job submit \\
                    --workflow-name {{PROJECT_NAME}}.workflow.llm_finetune_pipeline
            """
            validated = validate_dataset(s3_path=dataset)
            model_path = finetune_task(
                dataset_path=validated, model_name=model_name, epochs=epochs, batch_size=batch_size
            )
            return evaluate_perplexity(model_path=model_path)
    ''')

    config_yaml = textwrap.dedent(f"""\
        project: {project_name}
        image: {{{{ECR_REGISTRY}}}}/{{{{ECR_REPO}}}}/training-llm:1.1.0
        resources:
          head: {{cpu: "4", mem: "16Gi"}}
          workers: {{replicas: 2, cpu: "8", mem: "64Gi", gpu: "1"}}
    """)

    readme = textwrap.dedent(f"""\
        # {project_name}

        LLM fine-tuning project using LoRA on Ray distributed workers.

        **Image**: `training-llm` (ray-worker + transformers + peft + deepspeed)
        **Compute**: 2× GPU workers (A10G) via Ray on Flyte

        ## Quick Start

        ```bash
        # 1. Register the workflow
        mlp job register --source-dir . --project my-team --domain development

        # 2. Submit a training job
        mlp job submit --workflow-name {project_name}.workflow.llm_finetune_pipeline

        # 3. Monitor
        mlp job status --job-id <execution-id>
        mlp job logs --job-id <execution-id>
        ```

        ## Platform Resources

        - SDK docs: `projects/components/sdk/ml_platform_sdk/`
        - Examples: `examples/02_llm_finetune/`
        - Image layers: `projects/components/images/versions.env`
        - CLI reference: `docs/cli_reference.md`
    """)

    _write_files(
        project_dir,
        {
            "workflow.py": workflow_py,
            "config.yaml": config_yaml,
            "README.md": readme,
        },
    )
    return project_dir


def generate_rag_pipeline_project(project_name: str, base_dir: Path) -> Path:
    """Generate a RAG pipeline using data-cpu + genai-gpu images."""
    project_dir = base_dir / project_name

    workflow_py = textwrap.dedent('''\
        """
        RAG Indexing Pipeline

        DAG: [Chunk Documents] → [Generate Embeddings] → [Index into Vector Store]

        Images: data-cpu (chunking), genai-gpu (embeddings + indexing)
        """

        from typing import Tuple

        from flytekit import Resources, task, workflow

        DATA_IMAGE = "{{ECR_REGISTRY}}/{{ECR_REPO}}/data-cpu:1.2.0"
        GENAI_IMAGE = "{{ECR_REGISTRY}}/{{ECR_REPO}}/genai-gpu:1.1.0"


        @task(
            requests=Resources(cpu="2", mem="8Gi"),
            container_image=DATA_IMAGE,
            cache=True,
            cache_version="1.0",
        )
        def chunk_documents(
            s3_input_path: str,
            s3_output_path: str,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
        ) -> Tuple[str, int]:
            """Split documents into overlapping chunks for RAG indexing."""
            # Lazy imports
            import json
            import tempfile

            import boto3
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            s3 = boto3.client("s3")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            # Parse s3://bucket/prefix format
            if not s3_input_path.startswith("s3://") or "/" not in s3_input_path[5:]:
                raise ValueError(f"Invalid S3 path: {s3_input_path}. Expected s3://bucket/prefix")
            if not s3_output_path.startswith("s3://") or "/" not in s3_output_path[5:]:
                raise ValueError(f"Invalid S3 path: {s3_output_path}. Expected s3://bucket/prefix")
            src_bucket, src_prefix = s3_input_path[5:].split("/", 1)
            dst_bucket, dst_prefix = s3_output_path[5:].split("/", 1)

            paginator = s3.get_paginator("list_objects_v2")
            all_chunks = []
            for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
                for obj in page.get("Contents", []):
                    body = s3.get_object(Bucket=src_bucket, Key=obj["Key"])["Body"].read().decode()
                    all_chunks.extend(splitter.split_text(body))

            # Upload chunks as a single JSONL file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
                for chunk in all_chunks:
                    tmp.write(json.dumps({"text": chunk}) + "\\n")
                tmp_path = tmp.name

            s3.upload_file(tmp_path, dst_bucket, f"{dst_prefix}/chunks.jsonl")
            num_chunks = len(all_chunks)
            print(f"Chunked documents → {s3_output_path} ({num_chunks} chunks)")
            return s3_output_path, num_chunks


        @task(
            requests=Resources(cpu="4", mem="16Gi", gpu="1"),
            container_image=GENAI_IMAGE,
        )
        def generate_embeddings(
            chunks_s3_path: str,
            output_s3_path: str,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        ) -> str:
            """Generate embeddings for each chunk using sentence-transformers."""
            import json
            import tempfile

            import boto3
            import numpy as np
            from sentence_transformers import SentenceTransformer

            s3 = boto3.client("s3")
            model = SentenceTransformer(model_name)

            # Download chunks JSONL from S3
            if not chunks_s3_path.startswith("s3://") or "/" not in chunks_s3_path[5:]:
                raise ValueError(f"Invalid S3 path: {chunks_s3_path}. Expected s3://bucket/prefix")
            src_bucket, src_key = chunks_s3_path[5:].split("/", 1)
            chunks_key = f"{src_key}/chunks.jsonl" if not src_key.endswith(".jsonl") else src_key
            body = s3.get_object(Bucket=src_bucket, Key=chunks_key)["Body"].read().decode()
            texts = [json.loads(line)["text"] for line in body.splitlines() if line.strip()]

            # Encode in batches and save as .npy
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                np.save(tmp, embeddings)
                tmp_path = tmp.name

            dst_bucket, dst_prefix = output_s3_path[5:].split("/", 1)
            s3.upload_file(tmp_path, dst_bucket, f"{dst_prefix}/embeddings.npy")
            print(f"Embeddings saved to {output_s3_path} ({len(texts)} vectors)")
            return output_s3_path


        @task(
            requests=Resources(cpu="4", mem="16Gi"),
            container_image=GENAI_IMAGE,
        )
        def index_to_vector_store(
            embeddings_path: str,
            collection_name: str,
            backend: str = "pgvector",
        ) -> str:
            """Index embeddings into a vector database (pgvector/faiss/chromadb)."""
            print(f"Indexed into {backend} collection: {collection_name}")
            return collection_name


        @workflow
        def rag_indexing_pipeline(
            input_docs: str = "s3://{{S3_BUCKET}}/documents/",
            collection: str = "{{PROJECT_NAME}}_docs",
            chunk_size: int = 512,
        ) -> str:
            """RAG indexing: chunk → embed → index.

            Submit:
                mlp job submit \\
                    --workflow-name {{PROJECT_NAME}}.workflow.rag_indexing_pipeline
            """
            chunks_path, _ = chunk_documents(
                s3_input_path=input_docs,
                s3_output_path=f"s3://{{S3_BUCKET}}/chunks/{{PROJECT_NAME}}/",
                chunk_size=chunk_size,
            )
            embeddings_path = generate_embeddings(
                chunks_s3_path=chunks_path,
                output_s3_path=f"s3://{{S3_BUCKET}}/embeddings/{{PROJECT_NAME}}/",
            )
            return index_to_vector_store(
                embeddings_path=embeddings_path, collection_name=collection
            )
    ''')

    readme = textwrap.dedent(f"""\
        # {project_name}

        RAG indexing pipeline: chunk documents → embed → index into vector store.

        **Images**: `data-cpu` (chunking), `genai-gpu` (embeddings + indexing)
        **Components used**: text_chunker, vector_store_indexer patterns

        ## Quick Start

        ```bash
        mlp job register --source-dir . --project my-team --domain development
        mlp job submit --workflow-name {project_name}.workflow.rag_indexing_pipeline
        ```

        ## Platform Resources

        - Reusable components: `projects/components/components/genai/`
        - GenAI image: `genai-gpu` (vllm, langchain, sentence-transformers, chromadb, pgvector)
    """)

    _write_files(project_dir, {"workflow.py": workflow_py, "README.md": readme})
    return project_dir


def generate_distributed_training_project(project_name: str, base_dir: Path) -> Path:
    """Generate a distributed training project using Ray + ml-gpu image."""
    project_dir = base_dir / project_name

    workflow_py = textwrap.dedent('''\
        """
        Distributed GPU Training Pipeline

        DAG: [Prepare Data] → [Train on Ray (2+ GPUs)] → [Evaluate & Log to MLflow]

        Image: ml-gpu (flyte-gpu + transformers + accelerate + mlflow + scikit-learn)
        """

        from typing import Dict

        from flytekit import Resources, task, workflow
        from flytekitplugins.ray import HeadNodeConfig, RayJobConfig, WorkerNodeConfig

        ML_IMAGE = "{{ECR_REGISTRY}}/{{ECR_REPO}}/ml-gpu:1.1.1"

        ray_config = RayJobConfig(
            head_node_config=HeadNodeConfig(
                ray_start_params={"dashboard-host": "0.0.0.0"},
                requests=Resources(cpu="2", mem="8Gi"),
            ),
            worker_node_config=[
                WorkerNodeConfig(
                    group_name="gpu-workers",
                    replicas=2,
                    min_replicas=1,
                    max_replicas=4,
                    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
                )
            ],
        )


        @task(
            requests=Resources(cpu="2", mem="4Gi"),
            cache=True,
            cache_version="1.0",
        )
        def prepare_data(s3_path: str) -> str:
            """Download and validate training data from S3."""
            import boto3

            print(f"Preparing data from {s3_path}")
            return s3_path


        @task(
            task_config=ray_config,
            requests=Resources(cpu="2", mem="8Gi"),
            container_image=ML_IMAGE,
        )
        def train_distributed(data_path: str, config: Dict[str, int]) -> str:
            """Distributed training across Ray GPU workers.

            Each worker gets 1 GPU. Uses Ray Train TorchTrainer for DDP.
            """
            import ray.train
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer

            def train_loop(train_config):
                import mlflow
                import torch
                import torch.nn as nn
                from torch.optim import AdamW
                from torch.utils.data import DataLoader, TensorDataset

                rank = ray.train.get_context().get_world_rank()

                # Only rank 0 logs to MLflow
                if rank == 0:
                    mlflow.start_run()

                # Replace this model with your actual architecture
                input_dim = train_config.get("input_dim", 100)
                num_classes = train_config.get("num_classes", 10)
                model = nn.Linear(input_dim, num_classes)
                model = ray.train.torch.prepare_model(model)
                optimizer = AdamW(model.parameters(), lr=train_config.get("lr", 1e-3))
                criterion = nn.CrossEntropyLoss()

                # Replace with your actual DataLoader
                x = torch.randn(256, input_dim)
                y = torch.randint(0, num_classes, (256,))
                batch_sz = train_config.get("batch_size", 32)
                loader = DataLoader(TensorDataset(x, y), batch_size=batch_sz)
                loader = ray.train.torch.prepare_data_loader(loader)

                for epoch in range(train_config["epochs"]):
                    epoch_loss = 0.0
                    for batch_x, batch_y in loader:
                        optimizer.zero_grad()
                        loss = criterion(model(batch_x), batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    epoch_loss /= len(loader)
                    if rank == 0:
                        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    ray.train.report({"loss": epoch_loss, "epoch": epoch})

            trainer = TorchTrainer(
                train_loop_per_worker=train_loop,
                train_loop_config=config,
                scaling_config=ScalingConfig(
                    num_workers=2,
                    use_gpu=True,
                    resources_per_worker={"GPU": 1},
                ),
            )
            result = trainer.fit()
            return str(result.checkpoint)


        @task(requests=Resources(cpu="2", mem="4Gi"))
        def evaluate_model(model_path: str) -> float:
            """Evaluate model and log metrics to MLflow."""
            import mlflow

            accuracy = 0.0  # Replace with actual evaluation
            mlflow.log_metric("eval_accuracy", accuracy)
            print(f"Model accuracy: {accuracy}")
            return accuracy


        @workflow
        def training_pipeline(
            dataset: str = "s3://{{S3_BUCKET}}/data/train.parquet",
            epochs: int = 10,
            batch_size: int = 128,
        ) -> float:
            """Distributed training pipeline.

            Submit:
                mlp job submit \\
                    --workflow-name {{PROJECT_NAME}}.workflow.training_pipeline
            """
            data_path = prepare_data(s3_path=dataset)
            model_path = train_distributed(
                data_path=data_path, config={"epochs": epochs, "batch_size": batch_size}
            )
            return evaluate_model(model_path=model_path)
    ''')

    readme = textwrap.dedent(f"""\
        # {project_name}

        Distributed GPU training on Ray with MLflow experiment tracking.

        **Image**: `ml-gpu` (flyte-gpu + transformers + accelerate + mlflow + scikit-learn)
        **Compute**: 2× GPU workers (A10G) via Ray Train TorchTrainer

        ## Quick Start

        ```bash
        mlp job register --source-dir . --project my-team --domain development
        mlp job submit --workflow-name {project_name}.workflow.training_pipeline
        ```

        ## Platform Resources

        - SDK training task: `projects/components/sdk/ml_platform_sdk/tasks/training.py`
        - Example: `examples/05_distributed_training_ray/`
        - EFS for checkpoints: `ml_platform_sdk.tasks.efs`
    """)

    _write_files(project_dir, {"workflow.py": workflow_py, "README.md": readme})
    return project_dir


def generate_spark_etl_project(project_name: str, base_dir: Path) -> Path:
    """Generate a Spark ETL pipeline project."""
    project_dir = base_dir / project_name

    workflow_py = textwrap.dedent('''\
        """
        Spark ETL Pipeline

        DAG: [Extract & Clean (Spark)] → [Feature Engineering (Spark)] → [Train (Ray)]

        For datasets > 1 TB that need complex SQL transforms before training.
        Spark runs via the Spark-on-Kubernetes operator.
        """

        from flytekit import Resources, task, workflow
        from flytekitplugins.spark import Spark


        @task(
            task_config=Spark(
                spark_conf={
                    "spark.executor.instances": "10",
                    "spark.executor.memory": "8g",
                    "spark.executor.cores": "4",
                    "spark.driver.memory": "4g",
                    "spark.sql.adaptive.enabled": "true",
                }
            ),
            cache=True,
            cache_version="1.0",
        )
        def extract_and_clean(raw_data_path: str, output_path: str) -> str:
            """Extract raw data from S3, clean it, write cleaned Parquet."""
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            df = spark.read.parquet(raw_data_path)
            cleaned = df.dropna().dropDuplicates()
            cleaned.write.mode("overwrite").parquet(output_path)
            print(f"Cleaned {cleaned.count()} rows → {output_path}")
            return output_path


        @task(
            task_config=Spark(
                spark_conf={
                    "spark.executor.instances": "5",
                    "spark.executor.memory": "8g",
                }
            ),
            cache=True,
            cache_version="1.0",
        )
        def feature_engineering(cleaned_path: str, output_path: str) -> str:
            """Add features: text_length, word_count, etc."""
            from pyspark.sql import SparkSession
            from pyspark.sql import functions as F

            spark = SparkSession.builder.getOrCreate()
            df = spark.read.parquet(cleaned_path)
            features = df.withColumn("text_length", F.length("text")).withColumn(
                "word_count", F.size(F.split("text", " "))
            )
            features.write.mode("overwrite").parquet(output_path)
            print(f"Features saved → {output_path}")
            return output_path


        @workflow
        def etl_pipeline(
            raw_data: str = "s3://{{S3_BUCKET}}/raw/",
            cleaned_output: str = "s3://{{S3_BUCKET}}/cleaned/",
            features_output: str = "s3://{{S3_BUCKET}}/features/",
        ) -> str:
            """Spark ETL pipeline.

            Submit:
                mlp job submit \\
                    --workflow-name {{PROJECT_NAME}}.workflow.etl_pipeline
            """
            cleaned = extract_and_clean(raw_data_path=raw_data, output_path=cleaned_output)
            return feature_engineering(cleaned_path=cleaned, output_path=features_output)
    ''')

    readme = textwrap.dedent(f"""\
        # {project_name}

        Spark ETL pipeline for large-scale data processing (>1 TB).

        **Spark Operator**: Runs on Kubernetes via Spark-on-K8s operator
        **Caching**: Deterministic tasks are cached in S3

        ## Quick Start

        ```bash
        mlp job register --source-dir . --project my-team --domain development
        mlp job submit --workflow-name {project_name}.workflow.etl_pipeline
        ```

        ## Platform Resources

        - Example: `examples/03_spark_etl/`
        - Spark SDK: `ml_platform_sdk.tasks.spark`
    """)

    _write_files(project_dir, {"workflow.py": workflow_py, "README.md": readme})
    return project_dir


def generate_remote_execution_project(project_name: str, base_dir: Path) -> Path:
    """Generate a @remote execution project — zero-config GPU access."""
    project_dir = base_dir / project_name

    demo_py = textwrap.dedent('''\
        """
        Zero-Config Remote GPU Execution

        Use @remote to run GPU functions on the cluster without Docker builds,
        ECR pushes, or Flyte registration. Great for prototyping and one-off tasks.
        """

        from ml_platform_sdk.remote import remote


        @remote(gpu=1, memory="16Gi")
        def check_gpu():
            """Verify GPU access on the cluster."""
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"GPU: {device} ({mem:.1f} GB)")
                return {"device": device, "memory_gb": round(mem, 1)}
            return {"error": "No GPU available"}


        @remote(gpu=1, memory="32Gi", gpu_type="a10g")
        def quick_finetune(model_name: str = "meta-llama/Llama-3.1-8B"):
            """Quick LoRA fine-tune prototype — no Docker build needed."""
            import torch
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            param_count = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"Loaded {model_name} ({param_count:.1f}B params)")
            return {"model": model_name, "params_b": round(param_count, 1)}


        @remote(gpu=0, cpu="2", memory="4Gi")
        def process_data(s3_path: str):
            """CPU-only data processing on the cluster."""
            import boto3

            s3 = boto3.client("s3")
            print(f"Processing data from {s3_path}")
            return {"status": "done", "path": s3_path}


        if __name__ == "__main__":
            # Run these directly — they execute on the cluster
            print(check_gpu())
            print(quick_finetune())
    ''')

    readme = textwrap.dedent(f"""\
        # {project_name}

        Zero-config remote GPU execution using `@remote` decorator.

        No Docker builds, no ECR pushes, no Flyte registration.
        Functions are serialized via cloudpickle and run on cluster GPU pods.

        ## Quick Start

        ```bash
        # Just run the script — @remote handles everything
        python demo.py

        # Or import and call individual functions
        python -c "from demo import check_gpu; print(check_gpu())"
        ```

        ## How It Works

        1. Function + args serialized with cloudpickle
        2. Sent to the execution-service on the cluster
        3. Service spins up a GPU pod, executes, streams logs
        4. Result deserialized and returned locally

        ## Platform Resources

        - SDK: `ml_platform_sdk.remote`
        - Example: `examples/07_remote_execution/`
        - Execution service: `projects/components/services/execution-service/`
    """)

    _write_files(project_dir, {"demo.py": demo_py, "README.md": readme})
    return project_dir


def generate_notebook_project(project_name: str, base_dir: Path) -> Path:
    """Generate a notebook exploration project."""
    project_dir = base_dir / project_name

    readme = textwrap.dedent(f"""\
        # {project_name}

        Interactive Jupyter notebook for data exploration and prototyping.

        ## Quick Start

        ```bash
        # Launch JupyterHub (opens in browser)
        mlp notebook open

        # Or launch a standalone GPU pod for interactive work
        mlp pod launch --gpu 1 --image ml-gpu
        mlp pod connect <pod-name>
        ```

        ## Available Notebook Profiles

        | Profile                  | GPUs | RAM   | Instance     |
        |--------------------------|------|-------|------------- |
        | CPU Standard             | 0    | 8 GB  | m5.xlarge    |
        | Shared A10G (time-slice) | 1/4  | 4 GB  | g5 (shared)  |
        | Single A10G              | 1    | 16 GB | g5.xlarge    |
        | A10G High-Mem            | 1    | 128GB | g5.8xlarge   |
        | 4× A10G                  | 4    | 192GB | g5.12xlarge  |

        ## Pre-installed Libraries

        All GPU notebook profiles include PyTorch, transformers, accelerate,
        and have access to EFS shared storage at `/shared`.

        ## Connecting to Platform Services

        ```python
        import mlflow
        import os

        # MLflow experiment tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Shared EFS storage
        efs_path = "/shared"  # Persists across notebook restarts

        # S3 data access
        import boto3
        s3 = boto3.client("s3")
        ```

        ## Next Steps

        - Prototype in notebook, then convert to a Flyte workflow
        - Use `@remote(gpu=1)` for quick one-off GPU tasks
        - See `examples/04_notebook_workflow/` for patterns

        ## Platform Resources

        - JupyterHub config: `projects/jupyter/helm-values.yaml`
        - GPU idle monitor: auto-shuts down idle GPU notebooks
    """)

    _write_files(project_dir, {"README.md": readme})
    return project_dir


def substitute_template_vars(directory: Path, substitutions: dict):
    """Replace template variables in all files within a directory."""
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix in (".py", ".md", ".yaml", ".yml"):
            content = file_path.read_text()
            for key, value in substitutions.items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
            file_path.write_text(content)


# ═══════════════════════════════════════════════════════════════════════════
# Main Onboarding Command
# ═══════════════════════════════════════════════════════════════════════════

# All available project templates
PROJECT_TEMPLATES = {
    "1": (
        "Fine-tune a language model",
        "llm-finetune",
        generate_llm_finetune_project,
    ),
    "2": (
        "Train a model on multiple GPUs",
        "distributed-training",
        generate_distributed_training_project,
    ),
    "3": (
        "Build a RAG / document search pipeline",
        "rag-pipeline",
        generate_rag_pipeline_project,
    ),
    "4": (
        "Process large datasets (TB+)",
        "spark-etl",
        generate_spark_etl_project,
    ),
    "5": (
        "Run a GPU script (no Docker needed)",
        "remote-execution",
        generate_remote_execution_project,
    ),
    "6": (
        "Explore data / prototype in a notebook",
        "notebook-exploration",
        generate_notebook_project,
    ),
}


def extract_model_name(text: str) -> str:
    """
    Best-effort extraction of a model name / HuggingFace ID from free text.
    Falls back to 'meta-llama/Llama-3.1-8B' if nothing recognizable is found.
    """
    t = text.lower()

    # Known HuggingFace families
    known = [
        (r"llama[-\s]?3\.?1[-\s]?(\d+)b", "meta-llama/Llama-3.1-{s}B"),
        (r"llama[-\s]?3[-\s]?(\d+)b", "meta-llama/Llama-3-{s}B"),
        (r"llama[-\s]?(\d+)b", "meta-llama/Llama-{s}B"),
        (r"mistral[-\s]?(\d+)b", "mistralai/Mistral-{s}B-v0.1"),
        (r"mixtral[-\s]?(\d+)x(\d+)b", "mistralai/Mixtral-{s0}x{s1}B"),
        (r"falcon[-\s]?(\d+)b", "tiiuae/falcon-{s}b"),
        (r"gemma[-\s]?(\d+)b", "google/gemma-{s}b"),
        (r"qwen[-\s]?(\d+(?:\.\d+)?)b", "Qwen/Qwen-{s}B"),
    ]
    for pattern, template in known:
        m = re.search(pattern, t)
        if m:
            if "{s0}" in template:
                return template.format(s0=m.group(1), s1=m.group(2))
            return template.format(s=m.group(1))

    # Generic: extract a word + number + B pattern from original text (preserve casing)
    m = re.search(r"([\w][\w\-\.]*)[\s\-]+(\d+(?:\.\d+)?\s*[bB])", text)
    if m:
        name = m.group(1).strip()
        size = m.group(2).strip().upper().replace(" ", "")
        return f"{name}-{size}"

    return "meta-llama/Llama-3.1-8B"


def detect_intent(text: str) -> tuple:
    """Map a free-text description to a project template using keyword matching."""
    t = text.lower()
    if any(
        w in t
        for w in [
            "finetune",
            "fine-tune",
            "fine tune",
            "lora",
            "qlora",
            "llm",
            "language model",
            "llama",
            "gpt",
            "mistral",
            "bert",
            "instruction",
            "rlhf",
            "sft",
            "chat model",
            "instruct",
        ]
    ):
        return PROJECT_TEMPLATES["1"]
    if any(
        w in t
        for w in [
            "rag",
            "retriev",
            "semantic search",
            "embed",
            "vector",
            "document search",
            "knowledge base",
            "index",
        ]
    ):
        return PROJECT_TEMPLATES["3"]
    if any(
        w in t
        for w in [
            "spark",
            "etl",
            "large dataset",
            "terabyte",
            "petabyte",
            "sql transform",
            "data pipeline",
            "clean data",
            "tb data",
        ]
    ):
        return PROJECT_TEMPLATES["4"]
    if any(
        w in t
        for w in [
            "notebook",
            "explore",
            "jupyter",
            "prototype",
            "interactive",
            "visualiz",
            "analysis",
            "eda",
            "plot",
        ]
    ):
        return PROJECT_TEMPLATES["6"]
    if any(
        w in t
        for w in [
            "script",
            "remote",
            "quick run",
            "one-off",
            "no docker",
            "@remote",
            "test quickly",
        ]
    ):
        return PROJECT_TEMPLATES["5"]
    # Default: distributed training
    return PROJECT_TEMPLATES["2"]


def _load_config() -> tuple[dict, dict]:
    """Return (cluster_cfg, ecr_cfg), falling back to empty dicts if missing."""
    config_path = os.path.expanduser("~/.mlp/config.yaml")
    if not os.path.exists(config_path):
        return {}, {}
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("cluster", {}), cfg.get("ecr", {})


def _qa_loop(assistant: "BedrockAssistant", context: str) -> None:
    """
    Run the interactive Q&A loop.  Exits cleanly on Ctrl+C (ESC-equivalent)
    or when the user types 'done' / 'exit' / 'esc' / 'q'.
    """
    console.print(
        "[dim]Ask me anything. Press [bold]Ctrl+C[/bold] or type 'done' to exit wizard mode.[/dim]"
    )
    try:
        while True:
            try:
                question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            except KeyboardInterrupt:
                console.print("\n[dim]Exiting wizard. See you next time![/dim]")
                return

            if not question.strip() or question.lower() in (
                "done",
                "exit",
                "quit",
                "q",
                "no",
                "esc",
            ):
                console.print("[dim]Exiting wizard. See you next time![/dim]")
                return

            console.print("[dim]...[/dim]")
            answer = assistant.ask(question, context)
            console.print(Markdown(answer))
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting wizard. See you next time![/dim]")


def wizard():
    """
    Interactive onboarding wizard for new ML engineers.

    Guides you from zero to running a GPU training job in minutes.
    """
    # ── Fallback mode: launched automatically after an unknown command ────
    unknown_cmd = os.environ.pop("_MLPLAT_UNKNOWN_CMD", None)
    if unknown_cmd:
        console.print(
            Panel(
                f"[yellow]You tried: [bold]mlp {unknown_cmd}[/bold][/yellow]\n\n"
                "That's not a recognised command — here's what you probably want:",
                title="🤖 ML Platform Assistant",
                border_style="yellow",
            )
        )
        cluster, ecr = _load_config()
        assistant = BedrockAssistant(region=ecr.get("region", "us-west-2"))
        question = (
            f"The user tried to run: mlp {unknown_cmd}\n"
            "This command does not exist. In 2-3 sentences, tell them the correct "
            "mlp command(s) to achieve what they were trying to do, with examples."
        )
        console.print("[dim]...[/dim]")
        answer = assistant.ask(question)
        console.print(Markdown(answer))
        context = f"The user tried to run: mlp {unknown_cmd}"
        _qa_loop(assistant, context)
        return

    # ── Normal mode: project creation wizard ─────────────────────────────
    console.print(
        Panel(
            "[bold cyan]Hey! I'm your ML Platform assistant.[/bold cyan]\n\n"
            "Tell me what you want to build and I'll generate a\n"
            "ready-to-run project for you.",
            title="🚀 ML Platform",
            border_style="cyan",
        )
    )

    # ── Step 1: Config ───────────────────────────────────────────────────
    config_path = os.path.expanduser("~/.mlp/config.yaml")

    if not os.path.exists(config_path):
        console.print("\n[yellow]Looks like you haven't set up the CLI yet.[/yellow]")
        if Confirm.ask("Run initial setup now?"):
            from cli.commands.init import init

            init()
        else:
            console.print("[dim]Run 'mlp init' and come back anytime.[/dim]")
            raise typer.Exit(1)

    cluster, ecr = _load_config()

    # ── Step 2: Natural language intent ─────────────────────────────────
    console.print(
        "\n[dim]e.g. 'fine-tune Llama on my data', "
        "'train a classifier on multiple GPUs',\n"
        "     'build a document search system', "
        "'explore a new dataset in a notebook'[/dim]"
    )
    try:
        intent_text = Prompt.ask("\nWhat are you trying to build?")
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting wizard. See you next time![/dim]")
        return

    desc, default_name, generator_func = detect_intent(intent_text)

    console.print(f"\nGot it — sounds like a [bold]{desc}[/bold] project.")
    try:
        confirmed = Confirm.ask("Is that right?", default=True)
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting wizard. See you next time![/dim]")
        return

    if not confirmed:
        # Fallback: compact pick list
        console.print("\nNo problem — which of these fits best?")
        for key, (d, _, _) in PROJECT_TEMPLATES.items():
            console.print(f"  [cyan]{key}[/cyan]  {d}")
        try:
            choice = Prompt.ask(
                "\nEnter a number", choices=list(PROJECT_TEMPLATES.keys()), default="1"
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Exiting wizard. See you next time![/dim]")
            return
        desc, default_name, generator_func = PROJECT_TEMPLATES[choice]

    # ── Step 3: Project name ─────────────────────────────────────────────
    default_project_name = default_name.replace("-", "_")
    try:
        project_name = Prompt.ask(
            "\nWhat should we call this project?",
            default=default_project_name,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Exiting wizard. See you next time![/dim]")
        return
    project_name = project_name.replace("-", "_").replace(" ", "_").lower()

    output_dir = Path.cwd() / project_name

    if output_dir.exists():
        try:
            ok = Confirm.ask(
                f"\n[yellow]A folder named '{project_name}' already exists. Start fresh?[/yellow]"
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Exiting wizard. See you next time![/dim]")
            return
        if not ok:
            console.print("[dim]No problem — come back when ready.[/dim]")
            raise typer.Exit(1)

    console.print(f"\n[dim]Creating {project_name}/...[/dim]")
    project_dir = generator_func(project_name, Path.cwd())

    substitutions = {
        "PROJECT_NAME": project_name,
        "S3_BUCKET": cluster.get("s3_bucket", "ml-platform-data"),
        "ECR_REGISTRY": ecr.get("registry", ECR_REGISTRY),
        "ECR_REPO": ECR_REPO,
        "MODEL_NAME": extract_model_name(intent_text),
    }
    substitute_template_vars(project_dir, substitutions)

    # ── Step 4: Friendly completion ──────────────────────────────────────
    console.print(
        Panel(
            textwrap.dedent(f"""\
                [bold green]Your project is ready![/bold green]  [dim]{project_dir}[/dim]

                Open [bold]workflow.py[/bold], fill in your data path and model name,
                then run:

                   [dim]mlp job submit --project-dir {project_name}[/dim]

                To check on it:
                   [dim]mlp job status --job-id <id>[/dim]
                   [dim]mlp job logs   --job-id <id>[/dim]
            """),
            border_style="green",
        )
    )

    # ── Step 5: Open Q&A ─────────────────────────────────────────────────
    assistant = BedrockAssistant(region=ecr.get("region", "us-west-2"))
    context = (
        f"The user just created a '{desc}' project named '{project_name}'.\n"
        f"S3 bucket: {cluster.get('s3_bucket')}, ECR: {ecr.get('registry')}"
    )
    _qa_loop(assistant, context)
