"""
Data component — tokenize text data with prompt templates.

Image: ml-gpu
"""

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

# Predefined prompt templates
PROMPT_TEMPLATES = {
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ),
    "chat": ("<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{assistant}"),
}


@task(
    retries=2,
    requests=Resources(cpu="4", mem="16Gi"),
    limits=Resources(cpu="8", mem="32Gi"),
    cache=True,
    cache_version="1.0",
)
def tokenizer(
    raw_data: FlyteFile,
    model_id: str,
    prompt_template: str = "alpaca",
    max_length: int = 2048,
) -> FlyteFile:
    """Tokenize text data with prompt templates.

    Args:
        raw_data: JSONL file with raw text data.
        model_id: HuggingFace model ID (for tokenizer).
        prompt_template: Template name ("alpaca", "chat") or custom format string.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset in Arrow format.
    """
    import json
    import os

    from datasets import Dataset
    from transformers import AutoTokenizer

    # Download raw data
    raw_data.download()

    # Load tokenizer
    tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token

    # Load data
    texts = []
    with open(raw_data.path) as f:
        for line in f:
            item = json.loads(line.strip())

            # Apply prompt template
            if prompt_template in PROMPT_TEMPLATES:
                template = PROMPT_TEMPLATES[prompt_template]
                text = template.format(**item)
            elif prompt_template == "custom":
                # Assume item already has "text" field
                text = item.get("text", "")
            else:
                # Use provided template as format string
                text = prompt_template.format(**item)

            texts.append(text)

    # Create dataset
    dataset = Dataset.from_dict({"text": texts})

    # Tokenize
    def tokenize_function(examples):
        return tokenizer_obj(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Save to disk
    output_path = "/tmp/tokenized"
    tokenized.save_to_disk(output_path)

    # Create a tar archive for FlyteFile
    import tarfile

    archive_path = "/tmp/tokenized.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_path, arcname=os.path.basename(output_path))

    return FlyteFile(path=archive_path)
