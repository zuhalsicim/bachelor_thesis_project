# run_finetune_python.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- Configuration ---
# The base model from Hugging Face (NOT the GGUF version)
BASE_MODEL_NAME = "seeklhy/codes-7b" 
# The training data you already created
TRAIN_DATA_PATH = "./train_data/finetune_data.jsonl"
# Directory where the fine-tuned LoRA adapter will be saved
OUTPUT_DIR = "./models/lora-adapter-seeklhy-codes-7b"

def main():
    """
    Fine-tunes the seeklhy/codes-7b model using QLoRA (4-bit quantization).
    """
    print(" Starting Python-based fine-tuning process...")

    # --- 1. Load the Dataset ---
    print(f" Loading training data from: {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
    print(f" Dataset loaded with {len(dataset)} examples.")

    # --- 2. Configure Quantization (QLoRA) ---
    # This reduces VRAM usage significantly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(" 4-bit quantization configured (QLoRA).")

    # --- 3. Load Base Model and Tokenizer ---
    print(f" Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Automatically use the GPU
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(" Base model and tokenizer loaded.")

    # --- 4. Configure LoRA ---
    # Prepare the model for k-bit training and configure LoRA parameters
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear" # Apply LoRA to all linear layers
    )
    model = get_peft_model(model, lora_config)
    print(" LoRA configured.")

    # --- 5. Configure Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True, # Use mixed precision
        push_to_hub=False
    )

    # --- 6. Initialize the Trainer ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )
    print(" Trainer initialized. Starting fine-tuning...")
    
    # --- 7. Start Fine-Tuning ---
    trainer.train()
    
    print("\\n Fine-tuning complete!")
    print(f" LoRA adapter saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

