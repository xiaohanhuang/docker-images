"""Centralized AWS configuration.

All AWS-specific constants (account ID, region, ECR registry) are read
from environment variables with sensible defaults.  Every module should
import from here instead of hardcoding values.
"""

import os

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "805673386114")
ECR_REGISTRY = os.getenv(
    "ECR_REGISTRY",
    f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com",
)
