from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import config

print("Downloading processor...")
AutoProcessor.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    cache_dir=config.DATA_DIR
)

print("Downloading model...")
LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    cache_dir=config.DATA_DIR
)

print("Downloading MLLMU-Bench dataset...")
load_dataset(
    "franciscoliu/MLLMU-Bench",
    cache_dir=config.DATA_DIR
)

print("All downloads complete.")
