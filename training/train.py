"""
QLoRA Fine-tuning script for DialoGPT on mental health conversation data.

Uses:
- 4-bit quantization (QLoRA) for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- SFTTrainer from TRL for supervised fine-tuning
"""

import os
import sys
import logging
from typing import Optional

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, QLoRAConfig, TrainingConfig, DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_quantization_config(qlora_config: QLoRAConfig) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config for 4-bit loading."""
    compute_dtype = getattr(torch, qlora_config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_config.load_in_4bit,
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
    )

    return bnb_config


def load_base_model(
    model_config: ModelConfig,
    bnb_config: BitsAndBytesConfig,
) -> tuple:
    """Load the base DialoGPT model with 4-bit quantization."""
    logger.info(f"Loading model: {model_config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name,
        padding_side="left",
    )

    # DialoGPT uses eos_token as pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable cache for training

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def setup_lora(model, qlora_config: QLoRAConfig):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        r=qlora_config.lora_r,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        target_modules=qlora_config.target_modules,
        bias=qlora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied. Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model


def load_processed_dataset(data_config: DataConfig):
    """Load the preprocessed dataset from disk."""
    logger.info(f"Loading processed dataset from {data_config.processed_data_dir}")

    if not os.path.exists(data_config.processed_data_dir):
        raise FileNotFoundError(
            f"Processed dataset not found at {data_config.processed_data_dir}. "
            "Run data/preprocess.py first."
        )

    dataset = load_from_disk(data_config.processed_data_dir)
    logger.info(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

    return dataset


def create_training_args(train_config: TrainingConfig) -> TrainingArguments:
    """Create training arguments."""
    return TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        eval_steps=train_config.eval_steps,
        evaluation_strategy=train_config.evaluation_strategy,
        fp16=train_config.fp16,
        optim=train_config.optim,
        max_grad_norm=train_config.max_grad_norm,
        seed=train_config.seed,
        report_to=train_config.report_to,
        remove_unused_columns=False,
    )


def train(
    model_config: Optional[ModelConfig] = None,
    qlora_config: Optional[QLoRAConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    data_config: Optional[DataConfig] = None,
):
    """Run the full QLoRA fine-tuning pipeline."""
    # Use defaults if not provided
    if model_config is None:
        model_config = ModelConfig()
    if qlora_config is None:
        qlora_config = QLoRAConfig()
    if train_config is None:
        train_config = TrainingConfig()
    if data_config is None:
        data_config = DataConfig()

    # Setup quantization
    bnb_config = setup_quantization_config(qlora_config)

    # Load model and tokenizer
    model, tokenizer = load_base_model(model_config, bnb_config)

    # Apply LoRA
    model = setup_lora(model, qlora_config)

    # Load dataset
    dataset = load_processed_dataset(data_config)

    # Training arguments
    training_args = create_training_args(train_config)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=model_config.max_length,
        packing=False,
    )

    # Train
    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()

    # Save the final model (adapter only)
    final_model_path = os.path.join(train_config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Save training metrics
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")

    return model, tokenizer


if __name__ == "__main__":
    train()
