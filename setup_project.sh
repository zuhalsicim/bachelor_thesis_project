#!/bin/bash
# This script fully prepares the project environment after a fresh clone.

echo "--- Starting project setup ---"

# 1. Create Directories
echo "Creating directories for models and predictions..."
mkdir -p models
mkdir -p predictions

# 2. Download the LLM Model
echo "Downloading the LLM model file (this may take a while)..."
wget -P ./models wget -P ./models https://huggingface.co/TheBloke/OmniSQL-7B-GGUF/resolve/main/omnisql-7b.Q4_K_S.gguf

# 3. Compile llama.cpp
echo "Compiling the llama.cpp server with GPU support..."
cd external/llama.cpp
make LLAMA_CUBLAS=1
cd ../..  # Go back to the project root

# 4. Set Up Conda Environment
echo "Creating and setting up the Conda environment 'ehrsql_eval'..."
conda env create -f environment.yml

echo "--- Project setup finished! ---"
echo "To activate the environment, run: conda activate ehrsql_eval"