# -*- coding: utf-8 -*-
"""
Centralized configuration and hyperparameters for the ORPO Llama fine-tuning project.
"""

import torch
from datetime import datetime

# --- Training Parameters ---
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 6e-5
LR_WARMUP_STEPS = 100
CONTEXT_LENGTH = 1024
ALPHA_ORPO = 0.5
PROMPT_MAX_LENGTH = 512
COMPILE_MODEL = False
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
LOG_INTERVAL = 100
EVAL_INTERVAL = 5

# --- Model Hyperparameters ---
DROPOUT = 0.0
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 0.0

# --- System and Path Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging (Weights & Biases) ---
PROJECT_NAME = "orpo-llama-finetune"
ENABLE_WANDB_LOGGING = True
WANDB_PROJECT = PROJECT_NAME
WANDB_RUN_NAME = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# --- Dataset and Model Paths ---
DATASET_CACHE_PATH = "./dataset/orpo"
DATASET_HF_ID = "mlabonne/orpo-dpo-mix-40k"
TOKENIZER_PATH = "tokenizers/tok13456"
CHECKPOINT_DIR = './llm_models/'
DRIVE_FILE_ID = '1phEJZD4wD-MZuO0eKJgMPVeC299IbS4V'
OUTPUT_DIR = '.' # Directory to extract downloaded files
