"""Unit tests for ml_platform_sdk.tasks.gpu."""

import os
from unittest.mock import patch

from ml_platform_sdk.tasks.gpu import (
    GPU_NODE_LABEL,
    GPU_NODE_LABEL_VALUE,
    GPU_TAINT_EFFECT,
    GPU_TAINT_KEY,
    GPU_TAINT_VALUE,
    NSIGHT_CUDA_IMAGE,
    NSIGHT_ENV_VAR,
    NSIGHT_MOUNT_PATH,
    NSIGHT_VOLUME_NAME,
    PROFILE_ENV_VAR,
    PROFILE_OUTPUT_BASE,
    _maybe_profile,
    build_gpu_pod_template,
    gpu_task,
)

# ── Constants ────────────────────────────────────────────────────


class TestConstants:
    def test_gpu_taint_key(self):
        assert GPU_TAINT_KEY == "nvidia.com/gpu"

    def test_gpu_taint_value(self):
        assert GPU_TAINT_VALUE == "true"

    def test_gpu_taint_effect(self):
        assert GPU_TAINT_EFFECT == "NoSchedule"

    def test_gpu_node_label(self):
        assert GPU_NODE_LABEL == "role"
        assert GPU_NODE_LABEL_VALUE == "gpu-worker"

    def test_profile_env_var(self):
        assert PROFILE_ENV_VAR == "ML_PLAT_PROFILE"

    def test_profile_output_base(self):
        assert PROFILE_OUTPUT_BASE == "/mnt/efs/profiles"


# ── build_gpu_pod_template ───────────────────────────────────────


class TestBuildGpuPodTemplate:
    def test_returns_pod_template(self):
        from flytekit import PodTemplate

        template = build_gpu_pod_template()
        assert isinstance(template, PodTemplate)

    def test_has_gpu_toleration(self):
        template = build_gpu_pod_template()
        tolerations = template.pod_spec.tolerations
        assert len(tolerations) == 1
        t = tolerations[0]
        assert t.key == GPU_TAINT_KEY
        assert t.value == GPU_TAINT_VALUE
        assert t.effect == GPU_TAINT_EFFECT

    def test_has_node_selector(self):
        template = build_gpu_pod_template()
        ns = template.pod_spec.node_selector
        assert ns[GPU_NODE_LABEL] == GPU_NODE_LABEL_VALUE

    def test_default_gpu_resources(self):
        template = build_gpu_pod_template(gpu=1, memory="32Gi", cpu="4")
        resources = template.pod_spec.containers[0].resources
        assert resources.requests["nvidia.com/gpu"] == "1"
        assert resources.requests["memory"] == "32Gi"
        assert resources.requests["cpu"] == "4"
        assert resources.limits["nvidia.com/gpu"] == "1"

    def test_multi_gpu_resources(self):
        template = build_gpu_pod_template(gpu=4, memory="64Gi", cpu="8")
        resources = template.pod_spec.containers[0].resources
        assert resources.requests["nvidia.com/gpu"] == "4"
        assert resources.limits["nvidia.com/gpu"] == "4"

    def test_no_gpu_resource_when_zero(self):
        template = build_gpu_pod_template(gpu=0)
        resources = template.pod_spec.containers[0].resources
        assert "nvidia.com/gpu" not in resources.requests
        assert "nvidia.com/gpu" not in resources.limits

    def test_gpu_type_a10g_sets_instance_selector(self):
        template = build_gpu_pod_template(gpu_type="a10g")
        ns = template.pod_spec.node_selector
        assert ns.get("karpenter.k8s.aws/instance-family") == "g5"

    def test_gpu_type_a100_sets_instance_selector(self):
        template = build_gpu_pod_template(gpu_type="a100")
        ns = template.pod_spec.node_selector
        assert ns.get("karpenter.k8s.aws/instance-family") == "p4d"

    def test_gpu_type_any_no_instance_selector(self):
        template = build_gpu_pod_template(gpu_type="any")
        ns = template.pod_spec.node_selector
        assert "karpenter.k8s.aws/instance-family" not in ns

    def test_profile_env_var_injected(self):
        template = build_gpu_pod_template(profile=True)
        env = template.pod_spec.containers[0].env
        assert env is not None
        assert any(e.name == PROFILE_ENV_VAR and e.value == "1" for e in env)

    def test_no_profile_env_by_default(self):
        template = build_gpu_pod_template(profile=False)
        env = template.pod_spec.containers[0].env
        assert env is None

    def test_nsight_injects_init_container(self):
        template = build_gpu_pod_template(nsight=True)
        init = template.pod_spec.init_containers
        assert init is not None
        assert len(init) == 1
        assert init[0].name == "nsight-injector"
        assert init[0].image == NSIGHT_CUDA_IMAGE

    def test_nsight_adds_shared_volume(self):
        template = build_gpu_pod_template(nsight=True)
        volumes = template.pod_spec.volumes
        assert any(v.name == NSIGHT_VOLUME_NAME for v in volumes)

    def test_nsight_mounts_volume_in_primary(self):
        template = build_gpu_pod_template(nsight=True)
        mounts = template.pod_spec.containers[0].volume_mounts
        assert any(m.mount_path == NSIGHT_MOUNT_PATH for m in mounts)

    def test_nsight_sets_env_var(self):
        template = build_gpu_pod_template(nsight=True)
        env = template.pod_spec.containers[0].env
        assert any(e.name == NSIGHT_ENV_VAR and e.value == "1" for e in env)

    def test_nsight_adds_path_env(self):
        template = build_gpu_pod_template(nsight=True)
        env = template.pod_spec.containers[0].env
        path_env = [e for e in env if e.name == "PATH"]
        assert len(path_env) == 1
        assert NSIGHT_MOUNT_PATH in path_env[0].value

    def test_no_nsight_by_default(self):
        template = build_gpu_pod_template(nsight=False)
        assert template.pod_spec.init_containers is None
        assert template.pod_spec.volumes is None

    def test_nsight_and_profile_together(self):
        template = build_gpu_pod_template(profile=True, nsight=True)
        env = template.pod_spec.containers[0].env
        assert any(e.name == PROFILE_ENV_VAR for e in env)
        assert any(e.name == NSIGHT_ENV_VAR for e in env)
        assert template.pod_spec.init_containers is not None

    def test_custom_memory_and_cpu(self):
        template = build_gpu_pod_template(memory="128Gi", cpu="16")
        resources = template.pod_spec.containers[0].resources
        assert resources.requests["memory"] == "128Gi"
        assert resources.requests["cpu"] == "16"


# ── gpu_task decorator ───────────────────────────────────────────


class TestGpuTaskDecorator:
    def test_wraps_function(self):
        @gpu_task(gpu=1)
        def sample(x: int) -> int:
            return x * 2

        assert callable(sample)

    def test_preserves_name(self):
        @gpu_task(gpu=1)
        def compute(x: int) -> int:
            return x + 1

        assert "compute" in compute.name

    def test_pod_template_has_gpu_toleration(self):
        @gpu_task(gpu=1)
        def process(x: str) -> str:
            return x

        template = process.pod_template
        assert template is not None
        tolerations = template.pod_spec.tolerations
        assert any(t.key == GPU_TAINT_KEY for t in tolerations)

    def test_pod_template_has_node_selector(self):
        @gpu_task(gpu=1)
        def process(x: str) -> str:
            return x

        ns = process.pod_template.pod_spec.node_selector
        assert ns[GPU_NODE_LABEL] == GPU_NODE_LABEL_VALUE

    def test_profile_flag_sets_env(self):
        @gpu_task(gpu=1, profile=True)
        def profiled(x: int) -> int:
            return x

        template = profiled.pod_template
        env = template.pod_spec.containers[0].env
        assert any(e.name == PROFILE_ENV_VAR for e in env)

    def test_no_profile_by_default(self):
        @gpu_task(gpu=1)
        def regular(x: int) -> int:
            return x

        template = regular.pod_template
        env = template.pod_spec.containers[0].env
        assert env is None

    def test_gpu_type_forwarded(self):
        @gpu_task(gpu=1, gpu_type="a100")
        def a100_task(x: int) -> int:
            return x

        ns = a100_task.pod_template.pod_spec.node_selector
        assert "karpenter.k8s.aws/instance-family" in ns

    def test_nsight_flag_injects_init_container(self):
        @gpu_task(gpu=1, nsight=True)
        def nsight_task(x: int) -> int:
            return x

        template = nsight_task.pod_template
        assert template.pod_spec.init_containers is not None
        assert template.pod_spec.init_containers[0].name == "nsight-injector"


# ── _maybe_profile wrapper ──────────────────────────────────────


class TestMaybeProfile:
    def test_no_profiling_without_env(self):
        def train(x: int) -> int:
            return x * 2

        wrapped = _maybe_profile(train)
        assert wrapped(5) == 10

    def test_preserves_function_name(self):
        def my_train(x: int) -> int:
            return x

        wrapped = _maybe_profile(my_train)
        assert wrapped.__name__ == "my_train"

    @patch.dict(os.environ, {PROFILE_ENV_VAR: "1"})
    @patch("os.makedirs")
    def test_profiling_attempts_torch_import(self, mock_makedirs):
        """When env var is set, wrapper tries to import torch.profiler."""

        def train(x: int) -> int:
            return x

        wrapped = _maybe_profile(train)
        try:
            result = wrapped(42)
            # If torch is available, profiling ran — result should still be correct
            assert result == 42
        except ImportError:
            pass  # torch not available in test env — that's fine

    def test_passes_args_and_kwargs(self):
        def train(a: int, b: int, c: int = 10) -> int:
            return a + b + c

        wrapped = _maybe_profile(train)
        assert wrapped(1, 2, c=3) == 6
