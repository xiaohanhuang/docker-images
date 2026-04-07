import importlib.util
import os
import sys
from unittest.mock import MagicMock


def test_flink_workflow_definition():
    """Verify that the Flink workflow and task can be defined correctly."""
    # Mock flytekit and flytekitplugins.flink
    mock_flytekit = MagicMock()
    mock_flink = MagicMock()

    original_flytekit = sys.modules.get("flytekit")
    original_flink = sys.modules.get("flytekitplugins.flink")

    try:
        sys.modules["flytekit"] = mock_flytekit
        sys.modules["flytekitplugins.flink"] = mock_flink

        # Find the repo root
        # This test is in projects/components/sdk/tests/test_flink_workflow.py
        # So we go up 4 levels
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

        # Manually load the module from the path with digits
        module_name = "examples.09_flink_streaming.workflow"
        file_path = os.path.join(repo_root, "examples/09_flink_streaming/workflow.py")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load module {module_name!r} from path {file_path!r}: "
                f"spec={spec!r}, loader={None if spec is None else spec.loader!r}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Check that the workflow and task are defined
        assert module.real_time_pipeline is not None
        assert module.process_stream is not None

        print("Flink workflow definition verified via mock and dynamic import.")
    finally:
        if original_flytekit is not None:
            sys.modules["flytekit"] = original_flytekit
        elif "flytekit" in sys.modules:
            del sys.modules["flytekit"]

        if original_flink is not None:
            sys.modules["flytekitplugins.flink"] = original_flink
        elif "flytekitplugins.flink" in sys.modules:
            del sys.modules["flytekitplugins.flink"]


if __name__ == "__main__":
    test_flink_workflow_definition()
