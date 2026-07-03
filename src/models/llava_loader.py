import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


def load_llava_4bit(model_id: str = "llava-hf/llava-1.5-7b-hf"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    model.eval()

    return processor, model