# -*- coding: utf-8 -*-
"""
The ORPOTrainer class, which orchestrates the model fine-tuning process.
"""

import os
import math
from typing import Dict, List, Tuple, Union

# Third-party libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import transformers
from datasets import load_dataset, load_from_disk

# Local imports
import config
from model.llama import Llama, ModelArgs
from utils.gdrive import download_and_extract_zip

class ORPOTrainer:
    """Orchestrates the ORPO fine-tuning process."""
    def __init__(self):
        self.device = config.DEVICE
        self.wandb_logging = False
        print(f"INFO: Using device: {self.device}")

        # Setup components
        self._setup_logging()
        self.tokenizer = self._setup_tokenizer()
        self.model = self._setup_model()
        self.train_loader, self.val_loader = self._prepare_dataloaders()
        self.optimizer, self.scheduler = self._setup_optimizer()

    def _setup_logging(self):
        if config.ENABLE_WANDB_LOGGING:
            try:
                import wandb

                # Create a clean, serializable dictionary for wandb
                wandb_config = {
                    "batch_size": config.BATCH_SIZE,
                    "epochs": config.EPOCHS,
                    "learning_rate": config.LEARNING_RATE,
                    "lr_warmup_steps": config.LR_WARMUP_STEPS,
                    "context_length": config.CONTEXT_LENGTH,
                    "alpha_orpo": config.ALPHA_ORPO,
                    "compile_model": config.COMPILE_MODEL,
                    "dtype": str(config.DTYPE).split('.')[-1],  # Converts torch.bfloat16 to "bfloat16"
                    "dropout": config.DROPOUT,
                    "gradient_clip": config.GRADIENT_CLIP,
                    "weight_decay": config.WEIGHT_DECAY,
                    "device": config.DEVICE,
                    "dataset_hf_id": config.DATASET_HF_ID,
                }

                wandb.init(
                    project=config.WANDB_PROJECT, 
                    name=config.WANDB_RUN_NAME, 
                    config=wandb_config  # Pass the clean dictionary
                )
                self.wandb_logging = True
                print("INFO: Weights & Biases logging enabled.")
            except ImportError:
                print("WARNING: 'wandb' library not found. Disabling logging.")


    def _gdrive_download_if_needed(self):
        model_path = os.path.join(config.CHECKPOINT_DIR, "base.pt")
        if not os.path.exists(model_path) or not os.path.exists(config.TOKENIZER_PATH):
            print("INFO: Model or tokenizer not found. Downloading from Google Drive...")
            download_and_extract_zip(config.DRIVE_FILE_ID, config.OUTPUT_DIR)
            print("SUCCESS: Model and tokenizer downloaded.")
        else:
            print("INFO: Model and tokenizer found locally. Skipping download.")

    def _setup_tokenizer(self) -> transformers.PreTrainedTokenizer:
        self._gdrive_download_if_needed()
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _setup_model(self) -> Llama:
        print("INFO: Loading pretrained model checkpoint...")
        ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "base.pt"), weights_only=False)
        model_config = ckpt.pop("config")

        args = ModelArgs(
            dim=model_config.hidden_size, n_layers=model_config.num_hidden_layers,
            n_heads=model_config.num_attention_heads, n_kv_heads=model_config.num_key_value_heads,
            vocab_size=model_config.vocab_size, norm_eps=model_config.rms_norm_eps,
            rope_theta=model_config.rope_theta, max_seq_len=config.CONTEXT_LENGTH,
            dropout=config.DROPOUT, hidden_dim=model_config.intermediate_size,
            attention_bias=model_config.attention_bias, mlp_bias=model_config.mlp_bias
        )
        model = Llama(args)
        model.load_state_dict(ckpt)
        model = model.to(dtype=config.DTYPE).to(device=self.device)

        if config.COMPILE_MODEL:
            print("INFO: Compiling model with torch.compile()...")
            model = torch.compile(model)

        print(f"INFO: Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
        return model

    def _prepare_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        if os.path.exists(config.DATASET_CACHE_PATH):
            print(f"INFO: Loading pre-tokenized dataset from: {config.DATASET_CACHE_PATH}")
            dataset = load_from_disk(config.DATASET_CACHE_PATH)
        else:
            print(f"INFO: Loading dataset '{config.DATASET_HF_ID}' from Hugging Face Hub.")
            dataset = load_dataset(config.DATASET_HF_ID, split="all")
            dataset = dataset.filter(lambda r: r["source"] != "toxic-dpo-v0.2")

            def filter_by_prompt_length(ex: Dict) -> bool:
                prompt = self.tokenizer.apply_chat_template(ex['chosen'][:-1], tokenize=True, add_generation_prompt=True, return_tensors='pt')
                return prompt.size(-1) < config.PROMPT_MAX_LENGTH

            def preprocess(examples: Union[List, Dict]) -> Dict:
                prompts = [self.tokenizer.apply_chat_template(item[:-1], tokenize=False, add_generation_prompt=True) for item in examples['chosen']]
                chosen = [self.tokenizer.apply_chat_template(item, tokenize=False) for item in examples['chosen']]
                rejected = [self.tokenizer.apply_chat_template(item, tokenize=False) for item in examples['rejected']]

                model_in = self.tokenizer(prompts, max_length=config.CONTEXT_LENGTH, padding='max_length', truncation=True)
                pos_labels = self.tokenizer(chosen, max_length=config.CONTEXT_LENGTH, padding='max_length', truncation=True)
                neg_labels = self.tokenizer(rejected, max_length=config.CONTEXT_LENGTH, padding='max_length', truncation=True)

                model_in['positive_input_ids'] = pos_labels['input_ids']
                model_in['positive_attention_mask'] = pos_labels['attention_mask']
                model_in['negative_input_ids'] = neg_labels['input_ids']
                model_in['negative_attention_mask'] = neg_labels['attention_mask']
                return model_in

            print("INFO: Filtering and tokenizing dataset...")
            dataset = dataset.filter(filter_by_prompt_length)
            dataset = dataset.map(preprocess, batched=True, num_proc=os.cpu_count(), remove_columns=dataset.column_names)
            print(f"INFO: Saving processed dataset to: {config.DATASET_CACHE_PATH}")
            dataset.save_to_disk(config.DATASET_CACHE_PATH)

        dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.05)
        collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=config.BATCH_SIZE, collate_fn=collator, num_workers=2)
        val_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=config.BATCH_SIZE, collate_fn=collator, num_workers=2)

        print(f"INFO: Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")
        return train_loader, val_loader

    def _setup_optimizer(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.LEARNING_RATE,
            betas=(0.9, 0.95), eps=1e-8, fused=(self.device == 'cuda'),
            weight_decay=config.WEIGHT_DECAY
        )

        num_training_steps = len(self.train_loader) * config.EPOCHS

        def get_lr_schedule(current_step: int) -> float:
            warmup_steps = config.LR_WARMUP_STEPS
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_schedule)
        return optimizer, scheduler

    def _calculate_log_probs(self, logits, input_ids, prompt_mask, response_mask):
        answer_mask = response_mask[:, :-1] - prompt_mask[:, 1:]
        shifted_logits = logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]

        log_softmax = shifted_logits.log_softmax(dim=-1)
        token_indices = (answer_mask * shifted_labels).unsqueeze(2)
        per_token_logps = torch.gather(log_softmax, dim=2, index=token_indices).squeeze(2)

        dtype = config.DTYPE
        log_prob = torch.mul(per_token_logps, answer_mask.to(dtype)).sum(dim=1) / answer_mask.sum(dim=1).to(dtype)
        return log_prob.to(dtype)

    def _get_orpo_loss(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_mask = batch['attention_mask'] * batch['positive_attention_mask']

        # Positive (chosen) response loss
        pos_labels = batch['positive_input_ids'].clone()
        pos_labels[~prompt_mask.logical_not()] = self.tokenizer.pad_token_id
        pos_labels[pos_labels == self.tokenizer.pad_token_id] = -100
        outputs_pos, loss_pos = self.model(batch['positive_input_ids'], pos_labels)
        pos_prob = self._calculate_log_probs(outputs_pos, batch["positive_input_ids"], batch['attention_mask'], batch['positive_attention_mask'])

        # Negative (rejected) response loss
        neg_labels = batch['negative_input_ids'].clone()
        neg_labels[neg_labels == self.tokenizer.pad_token_id] = -100
        outputs_neg, _ = self.model(batch['negative_input_ids'], neg_labels)
        neg_prob = self._calculate_log_probs(outputs_neg, batch["negative_input_ids"], batch['attention_mask'], batch['negative_attention_mask'])

        # ORPO loss calculation
        log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
        log_sigmoid_ratio = F.logsigmoid(log_odds)

        total_loss = (loss_pos - config.ALPHA_ORPO * log_sigmoid_ratio).mean()
        return total_loss.to(config.DTYPE), log_odds.mean(), log_sigmoid_ratio.mean()

    @torch.no_grad()
    def _estimate_loss(self):
        self.model.eval()
        results = {}
        for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
            losses, log_odds, ratios = torch.zeros(config.EVAL_INTERVAL), torch.zeros(config.EVAL_INTERVAL), torch.zeros(config.EVAL_INTERVAL)
            iterator = iter(loader)
            for i in range(config.EVAL_INTERVAL):
                try:
                    batch = next(iterator)
                except StopIteration:
                    continue # Skip if loader is smaller than eval interval
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                loss, log_odd, ratio = self._get_orpo_loss(batch)
                losses[i], log_odds[i], ratios[i] = loss.item(), log_odd.item(), ratio.item()

            results[f'{split}/loss'] = losses.mean().item()
            results[f'{split}/log_odds'] = log_odds.mean().item()
            results[f'{split}/ratio_term'] = (config.ALPHA_ORPO * ratios.mean()).item()

        self.model.train()
        return results

    def _save_checkpoint(self, epoch_idx: int):
        state = {'config': self.model.params, **self.model.state_dict()}
        filename = f"{config.PROJECT_NAME}_epoch_{epoch_idx+1}.pt"
        path = os.path.join(config.CHECKPOINT_DIR, filename)
        torch.save(state, path)
        print(f"\nCheckpoint saved to {path}\n")

    def train(self):
        print("\n" + "="*50 + "\n S T A R T I N G   O R P O   T R A I N I N G\n" + "="*50 + "\n")
        try:
            initial_losses = self._estimate_loss()
            print(f"Initial Losses: {initial_losses}")

            for epoch in range(config.EPOCHS):
                self.model.train()
                progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")

                for i, batch in progress_bar:
                    self.optimizer.zero_grad(set_to_none=True)
                    batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                    total_loss, _, _ = self._get_orpo_loss(batch)

                    if torch.isnan(total_loss):
                        raise RuntimeError("ERROR: Loss is NaN. Aborting training.")

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.GRADIENT_CLIP)
                    self.optimizer.step()
                    self.scheduler.step()

                    if i > 0 and i % config.LOG_INTERVAL == 0:
                        eval_metrics = self._estimate_loss()
                        log_data = {**eval_metrics, "learning_rate": self.scheduler.get_last_lr()[0]}
                        print(f"\nStep: [{i}/{len(self.train_loader)}] | Metrics: {eval_metrics}")
                        if self.wandb_logging:
                            import wandb
                            wandb.log(log_data, step=(epoch * len(self.train_loader) + i))

                self._save_checkpoint(epoch)

        except (KeyboardInterrupt, RuntimeError) as e:
            print(f"\nTraining interrupted: {e}")
        finally:
            torch.cuda.empty_cache()
            if self.wandb_logging:
                import wandb
                wandb.finish()
            print("Cleanup complete. Training finished.")
