"""Official Qwen-Drive image encoding shared by engine workers and parity checks."""

def official_processor(tokenizer, original_config):
    from qwen_drive.configuration_qwen_drive import QwenDriveConfig
    from qwen_drive.scene import QwenDriveProcessor
    return QwenDriveProcessor(tokenizer, QwenDriveConfig(**original_config))


def encode_engine_input(processor, prompt, images):
    import torch
    from qwen_drive.scene import CameraFrame
    values = {'input_ids': [processor.tokenizer.encode(prompt, add_special_tokens=False)]}
    if images:
        patches, grids = [], []
        for image in images:
            patch, (rows, cols) = processor._patchify(CameraFrame(image), processor.config.current_image_pixels)
            patches.append(patch)
            grids.append((1, rows, cols))
        values.update(pixel_values=torch.cat(patches), image_grid_thw=torch.tensor(grids, dtype=torch.long))
    return values
