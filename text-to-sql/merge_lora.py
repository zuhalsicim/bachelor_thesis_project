# merge_lora_corrected.py
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# --- Configuration ---
BASE_MODEL_NAME = "seeklhy/codes-7b"
LORA_ADAPTER_PATH = "./models/lora-adapter-seeklhy-codes-7b/checkpoint-1755"
MERGED_MODEL_PATH = "./models/finetuned-seeklhy-codes-7b-merged"

def main():
    """
    Correctly merges a QLoRA adapter by loading the base model in 4-bit first.
    """
    print(" Starting Corrected LoRA Merge Process...")
    print(f"   - Base Model: {BASE_MODEL_NAME}")
    print(f"   - Adapter: {LORA_ADAPTER_PATH}")
    print(f"   - Output: {MERGED_MODEL_PATH}")

    # --- 1. Define the 4-bit Quantization Config (MUST match training) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("\\n 4-bit quantization config prepared.")

    # --- 2. Load the Base Model WITH Quantization ---
    print("\\n Loading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(" Base model and tokenizer loaded correctly.")

    # --- 3. Load the LoRA Adapter and Merge ---
    print("\\n Loading LoRA adapter onto quantized model...")
    # This now works because the model structures match
    merged_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # This performs the final merge
    merged_model = merged_model.merge_and_unload()
    print(" LoRA adapter merged successfully.")

    # --- 4. Save the Merged Model ---
    print(f"\\n Saving new merged model to: {MERGED_MODEL_PATH}")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("\\n Merge complete!")
    print("   Your new, fine-tuned model is ready for GGUF conversion.")

if __name__ == "__main__":
    main()
