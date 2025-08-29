# merge_lora.py
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
# The original base model from Hugging Face
BASE_MODEL_NAME = "Snowflake/Arctic-Text2SQL-R1-7B"
# The directory where your trained LoRA adapter was saved
LORA_ADAPTER_PATH = "./models/lora-adapter-arctic-ehrsql"
# The directory where the new, merged model will be saved
MERGED_MODEL_PATH = "./models/finetuned-arctic-7b-merged"

def main():
    """
    Merges a PEFT LoRA adapter with the base model and saves the new model.
    """
    print(" Starting LoRA merge process...")
    print(f"   - Base Model: {BASE_MODEL_NAME}")
    print(f"   - Adapter: {LORA_ADAPTER_PATH}")
    print(f"   - Output: {MERGED_MODEL_PATH}")

    # --- 1. Load the Base Model ---
    print("\\n Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    print(" Base model and tokenizer loaded.")

    # --- 2. Load the LoRA Adapter and Merge ---
    print("\\n Loading LoRA adapter and merging...")
    # This command loads the adapter and applies its changes to the base model
    merged_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    # This performs the final merge
    merged_model = merged_model.merge_and_unload()
    print(" LoRA adapter merged successfully.")

    # --- 3. Save the Merged Model ---
    print(f"\\n Saving new merged model to: {MERGED_MODEL_PATH}")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("\\ Merge complete")
    print("   Your new, fine-tuned model is ready.")

if __name__ == "__main__":
    main()
