"""
server.py — FastAPI inference server for the Text2SQL model.
Deployed via: make serve (creates a Kubernetes deployment)

Usage:
  POST /predict  {"question": "...", "context": "CREATE TABLE ..."}
  GET  /health
  GET  /model_info
"""
import os, boto3, tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import uvicorn

S3_BUCKET    = os.environ["S3_BUCKET"]
RUN_ID       = os.environ["MODEL_RUN_ID"]
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

app = FastAPI(
    title="Text2SQL Inference API",
    description="Fine-tuned Salesforce/codet5p-220m → SQL generation",
    version="1.0.0",
)

# ── Load model at startup ─────────────────────────────────────────────
print(f"🔄 Loading model from s3://{S3_BUCKET}/text2sql/checkpoints/{RUN_ID}")
_s3 = boto3.client("s3")
_ckpt_dir = tempfile.mkdtemp()

paginator = _s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"text2sql/checkpoints/{RUN_ID}/"):
    for obj in page.get("Contents", []):
        key   = obj["Key"]
        fname = key.split("/")[-1]
        body  = _s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        with open(f"{_ckpt_dir}/{fname}", "wb") as f:
            f.write(body)

_tokenizer = AutoTokenizer.from_pretrained(_ckpt_dir)
_model     = AutoModelForSeq2SeqLM.from_pretrained(_ckpt_dir)
_pipe      = pipeline(
    "text2text-generation",
    model=_model,
    tokenizer=_tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    device="cpu",
)
print("✅ Model loaded and ready")


# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    question: str
    context: str      # CREATE TABLE statements

class PredictResponse(BaseModel):
    sql: str
    prompt: str


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}


@app.get("/model_info")
def model_info():
    return {
        "model": "Salesforce/codet5p-220m",
        "run_id": RUN_ID,
        "s3_path": f"s3://{S3_BUCKET}/text2sql/checkpoints/{RUN_ID}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="context (CREATE TABLE) cannot be empty")

    prompt = f"translate English to SQL: {req.question} </s> Tables: {req.context}"
    output = _pipe(prompt)[0]["generated_text"].strip()
    return PredictResponse(sql=output, prompt=prompt)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
