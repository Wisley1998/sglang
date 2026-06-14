"""Unit tests for dLLM scheduler component behavior."""

from __future__ import annotations

import sys
import types
import unittest
from dataclasses import dataclass
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


config_module = types.ModuleType("sglang.srt.configs.model_config")
config_module.ModelConfig = object
configs_module = types.ModuleType("sglang.srt.configs")
configs_module.__path__ = []
configs_module.model_config = config_module

server_args_module = types.ModuleType("sglang.srt.server_args")
server_args_module.ServerArgs = object

req_module = types.ModuleType("sglang.srt.dllm.mixin.req")
req_module.DllmReqPhase = SimpleNamespace(
    STAGING_PREFILL="STAGING_PREFILL",
    INCOMING_PREFILL="INCOMING_PREFILL",
    STAGING_DECODE="STAGING_DECODE",
    INCOMING_DECODE="INCOMING_DECODE",
)

schedule_batch_module = types.ModuleType("sglang.srt.managers.schedule_batch")
schedule_batch_module.Req = object
schedule_batch_module.RequestStage = SimpleNamespace(PREFILL_WAITING="prefill_waiting")
schedule_batch_module.ScheduleBatch = object

schedule_policy_module = types.ModuleType("sglang.srt.managers.schedule_policy")
schedule_policy_module.AddReqResult = SimpleNamespace(
    CONTINUE="continue", NO_TOKEN="no_token"
)
schedule_policy_module.PrefillAdder = object

forward_batch_module = types.ModuleType("sglang.srt.model_executor.forward_batch_info")
forward_batch_module.ForwardMode = SimpleNamespace(DLLM_EXTEND="dllm_extend")

mem_cache_common_module = types.ModuleType("sglang.srt.mem_cache.common")
mem_cache_common_module.release_kv_cache = lambda *args, **kwargs: None

req_time_stats_module = types.ModuleType("sglang.srt.observability.req_time_stats")
req_time_stats_module.set_time_batch = lambda *args, **kwargs: None

_SCHEDULER_STUBS = {
    "sglang": types.ModuleType("sglang"),
    "sglang.srt": types.ModuleType("sglang.srt"),
    "sglang.srt.dllm": types.ModuleType("sglang.srt.dllm"),
    "sglang.srt.dllm.mixin": types.ModuleType("sglang.srt.dllm.mixin"),
    "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
    "sglang.srt.mem_cache": types.ModuleType("sglang.srt.mem_cache"),
    "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
    "sglang.srt.observability": types.ModuleType("sglang.srt.observability"),
    "sglang.srt.configs": configs_module,
    "sglang.srt.configs.model_config": config_module,
    "sglang.srt.server_args": server_args_module,
    "sglang.srt.dllm.mixin.req": req_module,
    "sglang.srt.managers.schedule_batch": schedule_batch_module,
    "sglang.srt.managers.schedule_policy": schedule_policy_module,
    "sglang.srt.mem_cache.common": mem_cache_common_module,
    "sglang.srt.model_executor.forward_batch_info": forward_batch_module,
    "sglang.srt.observability.req_time_stats": req_time_stats_module,
}


def _load_scheduler_components():
    sys.modules.pop("sglang.srt.dllm.config", None)
    sys.modules.pop("sglang.srt.dllm.mixin.scheduler", None)
    with patch.dict(sys.modules, _SCHEDULER_STUBS):
        config_spec = spec_from_file_location(
            "sglang.srt.dllm.config",
            REPO_ROOT / "python/sglang/srt/dllm/config.py",
        )
        config = module_from_spec(config_spec)
        sys.modules["sglang.srt.dllm.config"] = config
        config_spec.loader.exec_module(config)

        scheduler_spec = spec_from_file_location(
            "sglang.srt.dllm.mixin.scheduler",
            REPO_ROOT / "python/sglang/srt/dllm/mixin/scheduler.py",
        )
        scheduler = module_from_spec(scheduler_spec)
        sys.modules["sglang.srt.dllm.mixin.scheduler"] = scheduler
        scheduler_spec.loader.exec_module(scheduler)
    return config.DllmConfig, scheduler.DllmManager, scheduler.SchedulerDllmMixin


@dataclass
class FakeReq:
    rid: str
    done: bool = False

    def finished(self) -> bool:
        return self.done

    def is_dllm_prefill(self) -> bool:
        return False


class TestSchedulerDllmMixin(unittest.TestCase):
    def test_fetch_waiting_reqs_counts_waiting_and_staging_capacity(self):
        DllmConfig, DllmManager, SchedulerDllmMixin = _load_scheduler_components()

        cfg = DllmConfig(
            algorithm="JointThreshold",
            algorithm_config={},
            block_size=4,
            mask_id=151669,
            max_running_requests=4,
        )
        scheduler = SimpleNamespace(
            dllm_config=cfg,
            server_args=SimpleNamespace(max_running_requests=4),
            dllm_manager=DllmManager(cfg),
            waiting_queue=[FakeReq("new-0"), FakeReq("new-1"), FakeReq("new-2")],
        )
        scheduler.dllm_manager.waiting_queue = [FakeReq("managed-waiting")]
        scheduler.dllm_manager.staging_queue = [FakeReq("managed-staging")]

        SchedulerDllmMixin._fetch_waiting_reqs(scheduler)

        self.assertEqual(
            [req.rid for req in scheduler.dllm_manager.waiting_queue],
            ["managed-waiting", "new-0", "new-1"],
        )
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["new-2"])

    def test_fetch_waiting_reqs_ignores_finished_managed_reqs(self):
        DllmConfig, DllmManager, SchedulerDllmMixin = _load_scheduler_components()

        cfg = DllmConfig(
            algorithm="JointThreshold",
            algorithm_config={},
            block_size=4,
            mask_id=151669,
            max_running_requests=2,
        )
        scheduler = SimpleNamespace(
            dllm_config=cfg,
            server_args=SimpleNamespace(max_running_requests=2),
            dllm_manager=DllmManager(cfg),
            waiting_queue=[FakeReq("new-0"), FakeReq("new-1")],
        )
        scheduler.dllm_manager.waiting_queue = [FakeReq("finished-waiting", done=True)]
        scheduler.dllm_manager.staging_queue = [FakeReq("active-staging")]

        SchedulerDllmMixin._fetch_waiting_reqs(scheduler)

        self.assertEqual(
            [req.rid for req in scheduler.dllm_manager.waiting_queue],
            ["finished-waiting", "new-0"],
        )
        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["new-1"])


if __name__ == "__main__":
    unittest.main()
