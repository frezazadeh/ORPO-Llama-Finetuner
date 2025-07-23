# -*- coding: utf-8 -*-
"""
Main execution script for fine-tuning a Llama model with the ORPO algorithm.

This script initializes the ORPOTrainer and starts the training process
using the settings defined in config.py.
"""
import os

# --- Hugging Face API Token ---
# Set your Hugging Face token here to interact with the Hub.
# You can get a token from your Hugging Face account settings.
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

# --- Performance Enhancements & Debugging Options ---
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_printoptions(threshold=10000, sci_mode=False, precision=2)
# ---

from trainer import ORPOTrainer

def main():
    """
    Main function to initialize and run the ORPO trainer.
    """
    trainer = ORPOTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
