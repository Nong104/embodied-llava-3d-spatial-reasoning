import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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


def attach_lora(llava_model, r=8, lora_alpha=16, lora_dropout=0.05):
    """Attach LoRA adapters to the LLaVA language model's attention
    projections, implementing the second training stage described in
    Section 3.7. Only these small adapter matrices become trainable; the
    rest of the 7B backbone, including the vision tower, remains frozen.

    Existing code that calls llava_model.model(...), llava_model.lm_head(...),
    and llava_model.get_image_features(...) continues to work unchanged
    after this wrapping, since peft forwards attribute access down to the
    original model for anything it does not define itself.
    """
    llava_model = prepare_model_for_kbit_training(llava_model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )

    llava_model = get_peft_model(llava_model, lora_config)
    return llava_model