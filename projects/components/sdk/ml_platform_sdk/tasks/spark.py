"""
Spark Task Decorator for the ML Platform SDK.

Usage:
    from ml_platform_sdk.tasks.spark import spark_task

    @spark_task(
        spark_conf={"spark.executor.instances": "5", "spark.executor.memory": "8g"},
        hadoop_conf={"fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem"},
    )
    def preprocess_data(spark, input_path: str, output_path: str):
        df = spark.read.parquet(input_path)
        df_clean = df.dropna().filter(df["text"].isNotNull())
        df_clean.write.parquet(output_path, mode="overwrite")
"""

import functools
from typing import Callable, Dict, Optional

from flytekit import Resources, task
from flytekitplugins.spark import Spark


def spark_task(
    spark_conf: Optional[Dict[str, str]] = None,
    hadoop_conf: Optional[Dict[str, str]] = None,
    executor_instances: int = 2,
    executor_memory: str = "4g",
    executor_cores: int = 2,
    driver_memory: str = "2g",
    driver_cores: int = 1,
    cache_version: str = "1.0",
):
    """
    Decorator that wraps a function as a Flyte Spark Task.

    The decorated function receives a SparkSession as its first argument.

    Args:
        spark_conf: Additional Spark configuration key-value pairs.
        hadoop_conf: Hadoop configuration (e.g., S3 settings).
        executor_instances: Number of Spark executors.
        executor_memory: Memory per executor (e.g., "4g").
        executor_cores: CPU cores per executor.
        driver_memory: Memory for the Spark driver.
        driver_cores: CPU cores for the driver.
        cache_version: Flyte cache version string.
    """
    default_spark_conf = {
        "spark.driver.memory": driver_memory,
        "spark.driver.cores": str(driver_cores),
        "spark.executor.memory": executor_memory,
        "spark.executor.cores": str(executor_cores),
        "spark.executor.instances": str(executor_instances),
        "spark.dynamicAllocation.enabled": "false",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": (
            "com.amazonaws.auth.WebIdentityTokenCredentialsProvider"
        ),
    }

    if spark_conf:
        default_spark_conf.update(spark_conf)

    default_hadoop_conf = {
        "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    }
    if hadoop_conf:
        default_hadoop_conf.update(hadoop_conf)

    def decorator(fn: Callable):
        @functools.wraps(fn)
        @task(
            task_config=Spark(
                spark_conf=default_spark_conf,
                hadoop_conf=default_hadoop_conf,
            ),
            cache=True,
            cache_version=cache_version,
            requests=Resources(cpu=str(driver_cores), mem=driver_memory),
        )
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator
