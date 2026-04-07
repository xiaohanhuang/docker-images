"""Tests for the accelerate_task decorator and Platform helper."""

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torch")

from ml_platform_sdk.tasks.accelerate import (  # noqa: E402
    GPU_NODE_LABEL,
    GPU_NODE_LABEL_VALUE,
    GPU_TAINT_KEY,
    STRATEGY_ENV_VAR,
    Platform,
    _worker_fn,
    accelerate_task,
    platform,
)

# ── Constants ────────────────────────────────────────────────────


class TestConstants:
    def test_gpu_taint_key(self):
        assert GPU_TAINT_KEY == "nvidia.com/gpu"

    def test_gpu_node_label(self):
        assert GPU_NODE_LABEL == "role"
        assert GPU_NODE_LABEL_VALUE == "gpu-worker"

    def test_strategy_env_var(self):
        assert STRATEGY_ENV_VAR == "ACCELERATE_STRATEGY"


# ── Platform singleton ───────────────────────────────────────────


class TestPlatformSingleton:
    def test_platform_is_instance(self):
        assert isinstance(platform, Platform)


# ── Platform._heuristics ─────────────────────────────────────────


class TestHeuristics:
    def test_ddp_for_small_model(self):
        import torch

        model = torch.nn.Linear(10, 1)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 16 * 1024**3
            assert platform._heuristics(model) == "ddp"

    def test_deepspeed_for_large_model(self):
        import torch

        model = torch.nn.Linear(10000, 50000)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 1 * 1024**3
            assert platform._heuristics(model) == "deepspeed"

    def test_ddp_without_cuda(self):
        import torch

        model = torch.nn.Linear(10, 1)
        with patch("torch.cuda.is_available", return_value=False):
            assert platform._heuristics(model) == "ddp"

    def test_fsdp_for_medium_model(self):
        import torch

        model = torch.nn.Linear(1000, 5000)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 100 * 1024**2
            result = platform._heuristics(model)
            assert result in ("fsdp", "deepspeed")


# ── Platform.setup ────────────────────────────────────────────────


class TestPlatformSetup:
    def test_ddp_strategy_distributed(self):
        import torch

        model = MagicMock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(1, 1))]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with (
            patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp,
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}),
        ):
            mock_ddp.return_value = MagicMock()
            platform.setup(model, optimizer, strategy="ddp")
            mock_ddp.assert_called_once()

    def test_local_dev_skips_distributed(self):
        """When WORLD_SIZE is 1 (local dev), setup returns model unwrapped."""
        import torch

        model = MagicMock(spec=torch.nn.Module)
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with (
            patch("torch.nn.parallel.DistributedDataParallel") as mock_ddp,
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=False),
        ):
            m, _, _ = platform.setup(model, optimizer, strategy="ddp")
            mock_ddp.assert_not_called()
            assert m is model

    def test_dataloader_gets_distributed_sampler(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        model = MagicMock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(1, 1))]
        optimizer = MagicMock(spec=torch.optim.Optimizer)
        dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
        dataloader = DataLoader(dataset, batch_size=10)

        with (
            patch("torch.nn.parallel.DistributedDataParallel", return_value=MagicMock()),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}),
        ):
            _, _, new_dl = platform.setup(model, optimizer, dataloader, strategy="ddp")
            assert isinstance(new_dl.sampler, torch.utils.data.distributed.DistributedSampler)

    def test_strategy_env_var_overrides(self):
        import torch

        model = MagicMock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(1, 1))]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with (
            patch(
                "torch.nn.parallel.DistributedDataParallel", return_value=MagicMock()
            ) as mock_ddp,
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(os.environ, {STRATEGY_ENV_VAR: "ddp", "WORLD_SIZE": "2", "RANK": "0"}),
        ):
            # Even if strategy="auto", env var forces DDP
            platform.setup(model, optimizer, strategy="auto")
            mock_ddp.assert_called_once()


# ── _worker_fn ────────────────────────────────────────────────────


class TestWorkerFn:
    def test_sets_env_vars(self):
        fn = MagicMock(return_value="ok")

        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False):
            _worker_fn(fn, "ddp", 0, 1, "localhost", "29500", None, (), {})
            fn.assert_called_once()
            assert os.environ["MASTER_ADDR"] == "localhost"
            assert os.environ[STRATEGY_ENV_VAR] == "ddp"

    def test_captures_result_at_rank_0(self, tmp_path):
        result_file = str(tmp_path / "result.pkl")
        fn = MagicMock(return_value=42)

        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False):
            _worker_fn(fn, "ddp", 0, 1, "localhost", "29500", result_file, (), {})

        import cloudpickle

        with open(result_file, "rb") as f:
            assert cloudpickle.load(f) == 42

    def test_no_result_capture_at_non_rank_0(self, tmp_path):
        result_file = str(tmp_path / "result.pkl")
        fn = MagicMock(return_value=42)

        with patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=False):
            _worker_fn(fn, "ddp", 0, 2, "localhost", "29500", result_file, (), {})

        assert not os.path.exists(result_file)


# ── accelerate_task decorator ────────────────────────────────────


class TestAccelerateTask:
    def test_creates_flyte_task(self):
        @accelerate_task(num_nodes=2, gpus_per_node=4)
        def my_train(epochs: int):
            return epochs

        assert callable(my_train)
        assert my_train.task_config is not None
        assert my_train.task_config.worker.replicas == 1

    def test_resource_requests(self):
        @accelerate_task(gpus_per_node=4, cpu="8", mem="32Gi")
        def my_train(x: int) -> int:
            return x

        assert my_train.resources.requests.gpu == "4"
        assert my_train.resources.requests.cpu == "8"
        assert my_train.resources.requests.mem == "32Gi"

    def test_pod_template_has_gpu_toleration(self):
        @accelerate_task(gpus_per_node=1)
        def my_train(x: int) -> int:
            return x

        template = my_train.pod_template
        assert template is not None
        tolerations = template.pod_spec.tolerations
        assert any(t.key == GPU_TAINT_KEY for t in tolerations)

    def test_pod_template_has_node_selector(self):
        @accelerate_task(gpus_per_node=1)
        def my_train(x: int) -> int:
            return x

        ns = my_train.pod_template.pod_spec.node_selector
        assert ns[GPU_NODE_LABEL] == GPU_NODE_LABEL_VALUE

    def test_single_gpu_calls_worker_fn(self):
        @accelerate_task(gpus_per_node=1)
        def my_train(x: int) -> int:
            return x * 2

        with patch("ml_platform_sdk.tasks.accelerate._worker_fn", return_value=10) as mock_wf:
            result = my_train(x=5)
            mock_wf.assert_called_once()
            assert result == 10

    def test_multi_gpu_calls_mp_spawn(self):
        @accelerate_task(num_nodes=1, gpus_per_node=4)
        def my_train(epochs: int):
            return epochs

        with patch("torch.multiprocessing.spawn") as mock_spawn:
            with patch.dict(os.environ, {"RANK": "0"}):
                my_train(epochs=5)
                mock_spawn.assert_called_once()
                # Verify _mp_target (module-level function) is used as the target
                call_args = mock_spawn.call_args
                from ml_platform_sdk.tasks.accelerate import _mp_target

                assert call_args[0][0] is _mp_target

    def test_default_num_nodes(self):
        @accelerate_task()
        def my_train(x: int) -> int:
            return x

        # Default: 1 node → 0 workers
        assert my_train.task_config.worker.replicas == 0

    def test_rejects_invalid_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes must be >= 1"):
            accelerate_task(num_nodes=0)

    def test_rejects_invalid_gpus_per_node(self):
        with pytest.raises(ValueError, match="gpus_per_node must be >= 1"):
            accelerate_task(gpus_per_node=0)

    def test_multi_node_fails_without_master_addr(self):
        @accelerate_task(num_nodes=1, gpus_per_node=1)
        def my_train(x: int) -> int:
            return x

        with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}, clear=False):
            # Remove MASTER_ADDR if present
            os.environ.pop("MASTER_ADDR", None)
            with pytest.raises(RuntimeError, match="MASTER_ADDR"):
                my_train(x=1)


# ── Deepspeed ImportError ────────────────────────────────────────


class TestDeepspeedFallback:
    def test_raises_clear_error_when_deepspeed_missing(self):
        import torch

        model = MagicMock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(1, 1))]
        optimizer = MagicMock(spec=torch.optim.Optimizer)

        with (
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}),
            patch.dict("sys.modules", {"deepspeed": None}),
        ):
            with pytest.raises(ImportError, match="DeepSpeed is required"):
                platform.setup(model, optimizer, strategy="deepspeed")
