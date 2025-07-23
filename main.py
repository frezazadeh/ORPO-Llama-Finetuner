# -*- coding: utf-8 -*-
"""
Main execution script for fine-tuning a Llama model with the ORPO algorithm.

This script initializes the ORPOTrainer and starts the training process
using the settings defined in config.py.
"""

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
