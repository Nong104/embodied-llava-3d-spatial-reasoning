import gc
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM GB:", round(props.total_memory / 1024**3, 2))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)

print("Loading LLaVA 1.5 7B in 4bit...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

model.eval()

print("Loaded successfully.")
print("Model memory footprint GB:", round(model.get_memory_footprint() / 1024**3, 2))
print("CUDA allocated GB:", round(torch.cuda.memory_allocated() / 1024**3, 2))
print("CUDA reserved GB:", round(torch.cuda.memory_reserved() / 1024**3, 2))

del model
del processor
gc.collect()
torch.cuda.empty_cache()

print("4bit LLaVA load test passed.")