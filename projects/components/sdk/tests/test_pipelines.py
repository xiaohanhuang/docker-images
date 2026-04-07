import sys
from unittest.mock import MagicMock, patch

from ml_platform_sdk import (
    Pipeline,
    llm_sft_full_pipeline,
    llm_sft_lora_pipeline,
    text2sql_pipeline,
)


def test_pipeline_stubs_initialization():
    assert isinstance(llm_sft_lora_pipeline, Pipeline)
    assert llm_sft_lora_pipeline._stub_name == "pipeline.llm_sft_lora_pipeline"

    assert isinstance(llm_sft_full_pipeline, Pipeline)
    assert llm_sft_full_pipeline._stub_name == "pipeline.llm_sft_full_pipeline"

    assert isinstance(text2sql_pipeline, Pipeline)
    assert text2sql_pipeline._stub_name == "pipeline.text2sql_pipeline"


def test_pipeline_version_resolution():
    mock_remote = MagicMock()

    # Mock list_launch_plans_paginated
    mock_lp = MagicMock()
    mock_lp.id.version = "v123"
    mock_remote.client.list_launch_plans_paginated.return_value = ([mock_lp], None)

    # Ensure cli.utils is importable (it may not be installed in SDK-only envs)
    cli_utils_mock = MagicMock()
    cli_utils_mock.flyte_remote.return_value = mock_remote
    with patch.dict(sys.modules, {"cli": MagicMock(), "cli.utils": cli_utils_mock}):
        # Create a fresh pipeline instance to avoid cached versions
        from ml_platform_sdk.components import _resolved_versions

        _resolved_versions.clear()
        pipe = Pipeline(name="test.pipeline", inputs={"a": int}, outputs={"o0": str})

        assert pipe.reference.id.version == "unresolved"
        pipe._ensure_resolved()

        assert pipe.reference.id.version == "v123"
        mock_remote.client.list_launch_plans_paginated.assert_called_once()
        args, kwargs = mock_remote.client.list_launch_plans_paginated.call_args
        assert args[0].name == "test.pipeline"


def test_pipeline_getitem():
    pinned = text2sql_pipeline["v1.0.0"]
    assert isinstance(pinned, Pipeline)
    assert pinned.reference.id.version == "v1.0.0"
    assert pinned._stub_name == text2sql_pipeline._stub_name
