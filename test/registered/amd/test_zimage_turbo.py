"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""

import pytest

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionSamplingParams,
    DiffusionTestCase,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=1800, suite="nightly-amd-1-gpu-zimage-turbo", nightly=True)

AMD_ZIMAGE_CASES = [
    DiffusionTestCase(
        "zimage_image_t2i",
        DiffusionServerArgs(model_path="Tongyi-MAI/Z-Image-Turbo", modality="image"),
        DiffusionSamplingParams(
            prompt="Doraemon is eating dorayaki",
            output_size="1024x1024",
        ),
    ),
]


class TestZImageTurboAMD(DiffusionServerBase):
    """AMD nightly test for Z-Image-Turbo text-to-image generation."""

    @pytest.fixture(params=AMD_ZIMAGE_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param
