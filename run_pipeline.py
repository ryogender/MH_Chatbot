"""
End-to-end pipeline runner for the Mental Health Chatbot.

Usage:
    python run_pipeline.py --all              # Run full pipeline
    python run_pipeline.py --preprocess       # Data preprocessing only
    python run_pipeline.py --train            # Training only
    python run_pipeline.py --chat             # Launch chatbot only
    python run_pipeline.py --chat --no-adapter  # Launch without adapter
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preprocessing():
    """Run the data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("=" * 60)

    from data.preprocess import preprocess_pipeline
    from config import DataConfig

    config = DataConfig()
    dataset = preprocess_pipeline(config)

    logger.info(f"Preprocessing complete! Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
    return dataset


def run_training():
    """Run the QLoRA fine-tuning."""
    logger.info("=" * 60)
    logger.info("STEP 2: QLoRA Fine-tuning")
    logger.info("=" * 60)

    from training.train import train
    from config import ModelConfig, QLoRAConfig, TrainingConfig, DataConfig

    model, tokenizer = train(
        model_config=ModelConfig(),
        qlora_config=QLoRAConfig(),
        train_config=TrainingConfig(),
        data_config=DataConfig(),
    )

    logger.info("Training complete!")
    return model, tokenizer


def run_chatbot(use_adapter: bool = True, share: bool = False, port: int = 7860):
    """Launch the Gradio chatbot UI."""
    logger.info("=" * 60)
    logger.info("STEP 3: Launching Chatbot")
    logger.info("=" * 60)

    from app.chatbot_ui import initialize_chatbot, build_ui

    initialize_chatbot(use_adapter=use_adapter)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


def main():
    parser = argparse.ArgumentParser(
        description="Mental Health Chatbot - End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all              Run full pipeline (preprocess → train → chat)
  python run_pipeline.py --preprocess       Run data preprocessing only
  python run_pipeline.py --train            Run QLoRA fine-tuning only
  python run_pipeline.py --chat             Launch chatbot with fine-tuned model
  python run_pipeline.py --chat --no-adapter  Launch chatbot with base DialoGPT
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing")
    parser.add_argument("--train", action="store_true", help="Run QLoRA fine-tuning")
    parser.add_argument("--chat", action="store_true", help="Launch chatbot UI")
    parser.add_argument("--no-adapter", action="store_true", help="Run chatbot without fine-tuned adapter")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Chatbot server port")

    args = parser.parse_args()

    # Default to --all if no flags provided
    if not any([args.all, args.preprocess, args.train, args.chat]):
        logger.info("No flags specified. Use --help to see options.")
        logger.info("Running full pipeline (--all)...")
        args.all = True

    if args.all or args.preprocess:
        run_preprocessing()

    if args.all or args.train:
        run_training()

    if args.all or args.chat:
        use_adapter = not args.no_adapter
        run_chatbot(use_adapter=use_adapter, share=args.share, port=args.port)


if __name__ == "__main__":
    main()
