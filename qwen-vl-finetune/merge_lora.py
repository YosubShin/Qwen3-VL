from transformers import AutoModelForVision2Seq, AutoTokenizer
from peft import PeftModel
import torch

base_model = "Qwen/Qwen3-VL-8B-Instruct"
lora_path = "/home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct-lora/checkpoint-520"
merged_path = "/home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct-lora/merged"

print(f"Loading base model: {base_model}")
base = AutoModelForVision2Seq.from_pretrained(
    base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

print(f"Loading LoRA adapter: {lora_path}")
model = PeftModel.from_pretrained(base, lora_path)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to: {merged_path}")
model.save_pretrained(merged_path, safe_serialization=True)
AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(merged_path)

print("✅ Merge complete.")