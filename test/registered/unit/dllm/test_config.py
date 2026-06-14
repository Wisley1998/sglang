"""Unit tests for dLLM config parsing."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[4]
OVERLAY_PATH = REPO_ROOT / "sglang_fork_overlay"
if OVERLAY_PATH.exists():
    sys.path.insert(0, str(OVERLAY_PATH))
else:
    sys.path.insert(0, str(REPO_ROOT / "python"))


try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:

    def register_cpu_ci(*args, **kwargs):
        return None


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeModelConfig:
    hf_config = SimpleNamespace(architectures=["SDARForCausalLM"])

    @staticmethod
    def from_server_args(*args, **kwargs):
        return _FakeModelConfig()


model_config_module = types.ModuleType("sglang.srt.configs.model_config")
model_config_module.ModelConfig = _FakeModelConfig
configs_module = types.ModuleType("sglang.srt.configs")
configs_module.__path__ = []
configs_module.model_config = model_config_module

server_args_module = types.ModuleType("sglang.srt.server_args")
server_args_module.ServerArgs = object

_DLLM_CONFIG_STUBS = {
    "sglang": types.ModuleType("sglang"),
    "sglang.srt": types.ModuleType("sglang.srt"),
    "sglang.srt.dllm": types.ModuleType("sglang.srt.dllm"),
    "sglang.srt.configs": configs_module,
    "sglang.srt.configs.model_config": model_config_module,
    "sglang.srt.server_args": server_args_module,
}


def _load_dllm_config():
    sys.modules.pop("sglang.srt.dllm.config", None)
    with patch.dict(sys.modules, _DLLM_CONFIG_STUBS):
        spec = spec_from_file_location(
            "sglang.srt.dllm.config",
            REPO_ROOT / "python/sglang/srt/dllm/config.py",
        )
        module = module_from_spec(spec)
        sys.modules["sglang.srt.dllm.config"] = module
        spec.loader.exec_module(module)
        return module.DllmConfig


class TestDllmConfig(unittest.TestCase):
    def test_constructor_exposes_common_algorithm_config(self):
        DllmConfig = _load_dllm_config()

        cfg = DllmConfig(
            algorithm="JointThreshold",
            algorithm_config={"half_step": True, "yield_every": 4},
            block_size=32,
            mask_id=156895,
            max_running_requests=8,
        )

        self.assertTrue(cfg.half_step)
        self.assertEqual(cfg.yield_every, 4)

    def test_empty_yaml_algorithm_config_defaults_to_empty_dict(self):
        DllmConfig = _load_dllm_config()

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write("")
            config_file.flush()

            cfg = DllmConfig.from_server_args(
                SimpleNamespace(
                    dllm_algorithm="JointThreshold",
                    dllm_algorithm_config=config_file.name,
                    max_running_requests=None,
                    model_path="fake-model",
                    revision=None,
                )
            )

        self.assertEqual(cfg.algorithm_config, {})
        self.assertEqual(cfg.block_size, 4)
        self.assertEqual(cfg.mask_id, 151669)
        self.assertEqual(cfg.max_running_requests, 1)
        self.assertFalse(cfg.half_step)
        self.assertEqual(cfg.yield_every, 8)


if __name__ == "__main__":
    unittest.main()
