resource "aws_ecr_repository" "base_gpu" {
  name                 = "ml-platform/base-gpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "base_cpu" {
  name                 = "ml-platform/base-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── Layer 2: framework images ────────────────────────────────

resource "aws_ecr_repository" "flyte_cpu" {
  name                 = "ml-platform/flyte-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "flyte_gpu" {
  name                 = "ml-platform/flyte-gpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "ray_worker" {
  name                 = "ml-platform/ray-worker"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── Layer 3: workload images ─────────────────────────────────

resource "aws_ecr_repository" "data_cpu" {
  name                 = "ml-platform/data-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "ml_gpu" {
  name                 = "ml-platform/ml-gpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "genai_gpu" {
  name                 = "ml-platform/genai-gpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "training_llm" {
  name                 = "ml-platform/training-llm"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "notebook_cpu" {
  name                 = "ml-platform/notebook-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "notebook_custom" {
  name                 = "ml-platform/notebook-custom"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "notebook_marimo" {
  name                 = "ml-platform/notebook-marimo"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "spark_base" {
  name                 = "ml-platform/spark-base"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── Service / workflow images ────────────────────────────────

resource "aws_ecr_repository" "executor_pool" {
  name                 = "ml-platform/executor-pool"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "executor_pool_cpu" {
  name                 = "ml-platform/executor-pool-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "gpu_idle_monitor" {
  name                 = "ml-platform/gpu-idle-monitor"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "execution_service" {
  name                 = "ml-platform/execution-service"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "registry_service" {
  name                 = "ml-platform/registry-service"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "text2sql_serve" {
  name                 = "ml-platform/text2sql-serve"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "workflow_cpu" {
  name                 = "ml-platform/workflow-cpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "workflow_gpu" {
  name                 = "ml-platform/workflow-gpu"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "notebook_marimo_vscode" {
  name                 = "ml-platform/notebook-marimo-vscode"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}
