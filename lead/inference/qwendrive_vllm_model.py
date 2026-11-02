"""vLLM model plugin with Qwen-Drive's official CPU pixel preprocessing."""
from functools import cached_property

from transformers import BatchFeature
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor, Qwen3VLDummyInputsBuilder
from lead.inference.qwendrive_vllm_inputs import official_processor, encode_engine_input


class P7QwenDriveMultiModalProcessor(Qwen3VLMultiModalProcessor):
    @cached_property
    def drive_processor(self):
        cfg = self.info.get_hf_config()
        return official_processor(self.info.get_tokenizer(), cfg.p7_qwendrive_config)

    def _hf_processor_applies_updates(self, *args, **kwargs):
        # This processor leaves one placeholder per image; vLLM expands it using
        # the official image_grid_thw returned below.
        return False

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if mm_data.get('videos'):
            raise ValueError('P7 adapter accepts images only')
        images = mm_data.get('images', [])
        return BatchFeature(encode_engine_input(self.drive_processor, prompt, images))


@MULTIMODAL_REGISTRY.register_processor(
    P7QwenDriveMultiModalProcessor, info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class P7QwenDriveForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    pass
