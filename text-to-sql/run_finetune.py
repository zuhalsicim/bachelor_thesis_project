# run_finetune.py
from llama_cpp import Llama

# --- Configuration ---
# Path to the base model you want to fine-tune
BASE_MODEL_PATH = "./models/mradermacher/Arctic-Text2SQL-R1-7B-GGUF/Arctic-Text2SQL-R1-7B.Q4_K_S.gguf"
# Path to your prepared training data
TRAIN_FILE_PATH = "./train_data/finetune_data.jsonl"
# Path where the new, fine-tuned model will be saved
OUTPUT_MODEL_PATH = "./models/finetuned-Arctic-Text2SQL-R1-7B.Q4_K_S.gguf"

def main():
    """
    Loads a base GGUF model, fine-tunes it on the prepared dataset,
    and saves the new model.
    """
    print(" Initializing fine-tuning process...")
    
    # Load the base model
    llm = Llama(
        model_path=BASE_MODEL_PATH,
        n_gpu_layers=-1,  # Offload all layers to GPU
        n_ctx=4096,       # Context window size
        verbose=True
    )
    
    print(f" Base model loaded: {BASE_MODEL_PATH}")
    print(f" Training data: {TRAIN_FILE_PATH}")

    # Start the fine-tuning process
    # This will take a significant amount of time depending on your hardware.
    print("\\n Starting fine-tuning... This may take several hours.")
    llm.train(
        training_file=TRAIN_FILE_PATH,
        model_output_path=OUTPUT_MODEL_PATH,
        # --- Key Training Parameters ---
        n_epochs=3,              # Number of times to go through the dataset
        batch_size=4,            # Number of examples to process at once
        learning_rate=1e-4,      # How quickly the model learns
    )

    print(f"\\n Fine-tuning complete!")
    print(f"   New model saved to: {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
