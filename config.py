"""
Configuration file for the Mental Health Chatbot project.
Contains all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the base model."""
    model_name: str = "microsoft/DialoGPT-medium"
    tokenizer_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    device_map: str = "auto"


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    output_dir: str = "./outputs/mental-health-chatbot"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 25
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    fp16: bool = True
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    seed: int = 42
    report_to: str = "none"  # Set to "wandb" to enable W&B logging


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""
    empathetic_dialogues: str = "empathetic_dialogues"
    go_emotions: str = "google-research-datasets/go_emotions"
    counsel_chat: str = "nbertagnolli/counsel-chat"
    processed_data_dir: str = "./data/processed"
    max_conversation_length: int = 5  # Max turns in a conversation
    train_split: float = 0.9
    val_split: float = 0.1


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    adapter_path: str = "./outputs/mental-health-chatbot/final_model"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_beams: int = 1


@dataclass
class SafetyConfig:
    """Configuration for safety guardrails."""
    crisis_keywords: list = field(
        default_factory=lambda: [
            "suicide", "kill myself", "end my life", "want to die",
            "self-harm", "self harm", "cutting myself", "hurt myself",
            "no reason to live", "better off dead", "overdose",
            "jump off", "hang myself", "slit my wrists",
        ]
    )
    crisis_response: str = (
        "I'm really concerned about what you're sharing. "
        "Please reach out to a crisis helpline immediately:\n\n"
        "- **National Suicide Prevention Lifeline**: 988 (call or text)\n"
        "- **Crisis Text Line**: Text HOME to 741741\n"
        "- **International Association for Suicide Prevention**: "
        "https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "You are not alone, and there are people who care about you "
        "and want to help. Please reach out now."
    )
    disclaimer: str = (
        "**Disclaimer**: I am an AI chatbot designed to provide emotional "
        "support and empathetic conversation. I am NOT a licensed therapist "
        "or mental health professional. If you are in crisis or need "
        "professional help, please contact a qualified mental health provider "
        "or call emergency services."
    )


# Convenience function to get all configs
def get_configs():
    """Return all configuration objects."""
    return {
        "model": ModelConfig(),
        "qlora": QLoRAConfig(),
        "training": TrainingConfig(),
        "data": DataConfig(),
        "inference": InferenceConfig(),
        "safety": SafetyConfig(),
    }
