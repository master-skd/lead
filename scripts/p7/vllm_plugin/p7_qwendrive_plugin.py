def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        'P7QwenDriveForConditionalGeneration',
        'lead.inference.qwendrive_vllm_model:P7QwenDriveForConditionalGeneration',
    )
