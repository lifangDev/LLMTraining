import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from trl import SFTConfig
from argparse import ArgumentParser

import os
import json
from datasets import load_from_disk
from pathlib import Path
from typing import List, Dict

class MLFlowCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        for metric_name, metric_value in metrics.items():
            mflow.log_metric(metric_name, metric_value, step = state.global_step)

    def on_log(self, args, state, control, logs, **kwargs):
        if logs is not None:
            for key,value in logs.items():
                if isinstance(value, (int, float)):
                    mflow.log_metric(key, value, step = state.global_step)

def format_data(sample):

    prompt = sample["prompt"]
    flashcards = sample["flashcards"]

    return [
        {
            "role" : "system",
            "content" : "You are a helpful assistant"
        },
        {
            "role" : "user",
            "content" : prompt
        },
        {
            "role" : "assistant",
            "content" : flashcards
        },
    ]

def train(dataset_path : str, model_name : str, output_dir : str):

    dataset = load_from_disk(dataset_path)
    split_dataset = dataset.train_test_split(test_size = 0.2, seed = 42)

    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code = True,
        device_map = "auto",
        torch_dtype = torch.bfloat16,
        _attn_implementations = "sdqa",
    )

    model.save_pretrained("Qwen-0.5B.Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.save_pretrained("Qwen-0.5B-Instruct")

    def collate_fn(examples):
        texts = [tokenizer.apply_chat_template(example, tokenize=False) for example in examples]

        batch = tokenizer(texts, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
        learning_rate = 1e-4,
        lr_scheduler_type="constant",
        logging_steps=15,
    )