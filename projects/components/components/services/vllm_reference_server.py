"""
vLLM Reference Model Server for OpenRLHF.

This service hosts a frozen reference model (SFT checkpoint) using vLLM
for efficient batched inference during RLHF training. The reference model
is used to compute KL penalties to prevent the policy from drifting too
far from the supervised fine-tuning baseline.

Image: training-llm (with vllm)
"""

import argparse
import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceRequest(BaseModel):
    """Request for reference model log probabilities."""

    prompts: List[str]
    responses: List[str]
    max_length: int = 512


class ReferenceResponse(BaseModel):
    """Response with log probabilities from reference model."""

    log_probs: List[float]


app = FastAPI(title="vLLM Reference Model Server")

# Global engine instance
engine: Optional[AsyncLLMEngine] = None
model_path: str = ""


@app.on_event("startup")
async def startup_event():
    """Initialize vLLM engine on startup."""
    global engine, model_path
    logger.info(f"Initializing vLLM engine with model: {model_path}")

    # Parse engine args
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=False,  # Use CUDA graphs for better performance
        disable_log_stats=False,
    )

    # Check for tensor parallelism from env
    tensor_parallel_size = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    if tensor_parallel_size > 1:
        engine_args.tensor_parallel_size = tensor_parallel_size
        logger.info(f"Using tensor parallelism: {tensor_parallel_size}")

    # Create engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM engine initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global engine
    if engine:
        # vLLM doesn't require explicit cleanup
        logger.info("Shutting down vLLM engine")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "healthy", "model": model_path}


@app.post("/reference", response_model=ReferenceResponse)
async def compute_reference_log_probs(request: ReferenceRequest) -> ReferenceResponse:
    """Compute log probabilities from reference model.

    Args:
        request: ReferenceRequest with prompts and responses

    Returns:
        ReferenceResponse with log probabilities for each response
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if len(request.prompts) != len(request.responses):
        raise HTTPException(
            status_code=400,
            detail=f"Prompts and responses must have same length: "
            f"{len(request.prompts)} != {len(request.responses)}",
        )

    try:
        # Combine prompts and responses
        full_texts = [p + r for p, r in zip(request.prompts, request.responses)]

        # Create sampling params for log prob computation
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=0,  # We only want log probs, not generation
            logprobs=1,  # Request log probs
        )

        # Generate with vLLM
        results = []
        for text in full_texts:
            result = await engine.generate(text, sampling_params, request_id=None)
            results.append(result)

        # Extract log probs from results
        log_probs = []
        for result in results:
            if result.outputs:
                # Sum log probs across tokens
                output = result.outputs[0]
                if hasattr(output, "cumulative_logprob"):
                    log_probs.append(float(output.cumulative_logprob))
                else:
                    # Fallback: sum token log probs
                    token_logprobs = [lp for lp in output.logprobs if lp is not None]
                    log_probs.append(float(sum(token_logprobs)))
            else:
                log_probs.append(0.0)

        return ReferenceResponse(log_probs=log_probs)

    except Exception as e:
        logger.error(f"Error computing reference log probs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the vLLM reference model server."""
    parser = argparse.ArgumentParser(description="vLLM Reference Model Server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the reference model (SFT checkpoint)",
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
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers",
    )

    args = parser.parse_args()

    global model_path
    model_path = args.model_path

    # Validate model path
    if not os.path.exists(model_path) and not model_path.startswith("s3://"):
        logger.warning(f"Model path does not exist locally: {model_path}")

    # Run server
    import uvicorn

    logger.info(f"Starting vLLM Reference Model Server on {args.host}:{args.port}")
    logger.info(f"Model: {model_path}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
