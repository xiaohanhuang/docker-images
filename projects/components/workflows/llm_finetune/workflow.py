from flytekit import workflow
from flytekit.types.directory import FlyteDirectory

# Import from our SDK
from ml_platform_sdk.tasks.data import download_dataset
from ml_platform_sdk.tasks.training import train_ray_task


@workflow
def llm_finetune_workflow(
    s3_dataset_path: str = "s3://my-bucket/dataset",
    num_epochs: int = 3,
    batch_size: int = 4,
) -> FlyteDirectory:
    """
    End-to-end workflow for finetuning an LLM using Ray.
    """

    # 1. Download Data
    dataset_file = download_dataset(s3_path=s3_dataset_path)

    # 2. Train Distributed
    # We pass the local path of the downloaded file (handled by Flyte)
    # and a config dict.
    training_config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "model_name": "meta-llama/Llama-2-7b-hf",
    }

    model_checkpoint = train_ray_task(
        dataset_path=dataset_file.path, training_config=training_config
    )

    return model_checkpoint
