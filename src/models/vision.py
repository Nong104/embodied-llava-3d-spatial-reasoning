import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def encode_image(frame, processor, llava_model, device):
    """Encode a real RGB frame into image tokens using LLaVA's own frozen
    CLIP vision tower and multimodal projector, exactly as specified in
    Section 3.3.1. This gives tokens that reflect real visual content and
    are already aligned to the language model's embedding space, replacing
    the random placeholder used earlier in development.

    frame can be a numpy array such as AI2-THOR's event.frame, or a PIL
    Image directly.
    """
    if isinstance(frame, np.ndarray):
        image = Image.fromarray(frame)
    else:
        image = frame

    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"]
    model_dtype = next(llava_model.parameters()).dtype
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    image_outputs = llava_model.get_image_features(pixel_values=pixel_values)
    image_tokens = image_outputs.pooler_output[0].unsqueeze(0)

    return image_tokens