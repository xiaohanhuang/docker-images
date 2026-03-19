"""
Reward Model Server for OpenRLHF.

This service hosts a reward model for scoring responses during RLHF training.
Supports two backends:
1. vLLM: For large reward models with efficient batching
2. Transformers: For smaller models with custom logic

The server exposes a gRPC endpoint for scoring responses, which can be called
by the OpenRLHF actor during rollout generation.

Image: training-llm
"""

import argparse
import logging
import os
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardRequest(BaseModel):
    """Request for reward scoring."""

    texts: List[str]


class RewardResponse(BaseModel):
    """Response with reward scores."""

    rewards: List[float]


app = FastAPI(title="Reward Model Server")

# Global model and tokenizer
reward_model: Optional[torch.nn.Module] = None
tokenizer = None
model_path: str = ""
backend: str = "transformers"
device: str = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def startup_event():
    """Initialize reward model on startup."""
    global reward_model, tokenizer, model_path, backend, device

    logger.info(f"Initializing reward model: {model_path}")
    logger.info(f"Backend: {backend}")
    logger.info(f"Device: {device}")

    if backend == "vllm":
        # vLLM backend for large models
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=1024,
            gpu_memory_utilization=0.9,
        )

        # Check for tensor parallelism from env
        tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        if tensor_parallel_size > 1:
            engine_args.tensor_parallel_size = tensor_parallel_size
            logger.info(f"Using tensor parallelism: {tensor_parallel_size}")

        reward_model = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM reward model initialized")

    else:
        # Transformers backend
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Load model and tokenizer
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Move to device
        reward_model = reward_model.to(device)
        reward_model.eval()

        logger.info(f"Transformers reward model initialized on {device}")

    logger.info("Reward model server ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down reward model server")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if reward_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model": model_path, "backend": backend}


@app.post("/reward", response_model=RewardResponse)
async def compute_rewards(request: RewardRequest) -> RewardResponse:
    """Compute reward scores for given texts.

    Args:
        request: RewardRequest with texts to score

    Returns:
        RewardResponse with reward scores
    """
    if reward_model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        if backend == "vllm":
            # vLLM backend: use generation with reward head
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=0,  # We only want the reward score
            )

            results = []
            for text in request.texts:
                result = await reward_model.generate(text, sampling_params, request_id=None)
                results.append(result)

            # Extract rewards (assuming reward is in logits)
            rewards = []
            for result in results:
                if result.outputs:
                    # Extract reward from output
                    output = result.outputs[0]
                    if hasattr(output, "reward"):
                        rewards.append(float(output.reward))
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)

        else:
            # Transformers backend
            inputs = tokenizer(
                request.texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reward_model(**inputs)
                logits = outputs.logits.squeeze(-1)  # (B,)
                rewards = logits.cpu().float().tolist()

        return RewardResponse(rewards=rewards)

    except Exception as e:
        logger.error(f"Error computing rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the reward model server."""
    parser = argparse.ArgumentParser(description="Reward Model Server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the reward model checkpoint",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend for model inference",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers",
    )

    args = parser.parse_args()

    global model_path, backend
    model_path = args.model_path
    backend = args.backend

    # Validate model path
    if not os.path.exists(model_path) and not model_path.startswith("s3://"):
        logger.warning(f"Model path does not exist locally: {model_path}")

    # Run server
    import uvicorn

    logger.info(f"Starting Reward Model Server on {args.host}:{args.port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Backend: {backend}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
