"""
Data preprocessing script for the Mental Health Chatbot.

Loads and processes:
1. EmpatheticDialogues (Facebook) - empathetic conversation pairs
2. GoEmotions (Reddit/Google) - emotion-labeled Reddit comments
3. CounselChat - real counselor-patient Q&A pairs

Outputs a unified dataset formatted for DialoGPT fine-tuning.
"""

import os
import json
import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# Add parent directory to path for config import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_empathetic_dialogues() -> list[dict]:
    """
    Load and process the EmpatheticDialogues dataset.
    Extracts conversation pairs (context -> response) with emotion labels.
    """
    logger.info("Loading EmpatheticDialogues dataset...")
    dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)

    conversations = []
    for split in ["train", "validation", "test"]:
        data = dataset[split]
        current_conv_id = None
        current_turns = []

        for row in data:
            conv_id = row["conv_id"]
            utterance = row["utterance"].strip()
            emotion = row["context"]  # emotion label

            # Clean up utterance (remove _comma_ tokens etc.)
            utterance = utterance.replace("_comma_", ",").strip()

            if conv_id != current_conv_id:
                # Save previous conversation
                if len(current_turns) >= 2:
                    for i in range(0, len(current_turns) - 1, 2):
                        if i + 1 < len(current_turns):
                            conversations.append({
                                "input": current_turns[i],
                                "response": current_turns[i + 1],
                                "emotion": emotion,
                                "source": "empathetic_dialogues",
                            })
                current_conv_id = conv_id
                current_turns = [utterance]
            else:
                current_turns.append(utterance)

        # Don't forget the last conversation
        if len(current_turns) >= 2:
            for i in range(0, len(current_turns) - 1, 2):
                if i + 1 < len(current_turns):
                    conversations.append({
                        "input": current_turns[i],
                        "response": current_turns[i + 1],
                        "emotion": emotion if emotion else "neutral",
                        "source": "empathetic_dialogues",
                    })

    logger.info(f"Extracted {len(conversations)} conversation pairs from EmpatheticDialogues")
    return conversations


def load_go_emotions() -> list[dict]:
    """
    Load and process the GoEmotions dataset (Reddit comments with emotion labels).
    Creates empathetic response templates based on detected emotions.
    """
    logger.info("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)

    # Emotion label mapping
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]

    # Empathetic response templates for mental-health-relevant emotions
    response_templates = {
        "sadness": [
            "I hear you, and it's okay to feel sad. Would you like to talk more about what's on your mind?",
            "That sounds really difficult. I'm here to listen whenever you need to share.",
            "I'm sorry you're going through this. Your feelings are completely valid.",
        ],
        "fear": [
            "It's understandable to feel afraid. What specifically is causing you the most worry right now?",
            "Fear can be overwhelming. Let's try to break down what's bothering you together.",
            "I can see this is scary for you. Remember, it's brave to acknowledge your fears.",
        ],
        "anger": [
            "I can understand why you'd feel angry about that. It's a natural response.",
            "Your frustration makes sense. Would it help to talk through what triggered this feeling?",
            "It's okay to feel angry. Let's explore what's beneath that anger together.",
        ],
        "nervousness": [
            "Feeling nervous is completely normal. What's making you feel this way?",
            "I understand that anxiety can be really tough. Let's take this one step at a time.",
            "It's okay to feel nervous. Would you like to try some grounding techniques together?",
        ],
        "disappointment": [
            "I'm sorry things didn't work out as you hoped. That must be really frustrating.",
            "Disappointment is hard to deal with. What were you hoping would happen?",
            "I understand your disappointment. Sometimes things don't go as planned, and that's tough.",
        ],
        "grief": [
            "I'm so sorry for your loss. Grief is one of the hardest things to go through.",
            "There's no right or wrong way to grieve. I'm here for you.",
            "Losing someone is incredibly painful. Take all the time you need.",
        ],
        "embarrassment": [
            "Please don't be too hard on yourself. Everyone has moments like this.",
            "I understand that feeling. It takes courage to share something embarrassing.",
            "That must have been uncomfortable, but it doesn't define who you are.",
        ],
        "remorse": [
            "It shows a lot of character that you feel remorseful. That means you care.",
            "Feeling guilty is tough, but it also shows you have a strong moral compass.",
            "It's okay to feel regret. What matters is how you move forward from here.",
        ],
        "confusion": [
            "It's okay to feel confused. Let's try to work through this together.",
            "Feeling lost is part of being human. What specifically is confusing you?",
            "I understand the confusion. Sometimes taking a step back helps gain perspective.",
        ],
        "love": [
            "It's wonderful that you have love in your life. Tell me more about it.",
            "Love is such a powerful emotion. How does it make you feel overall?",
        ],
        "joy": [
            "That's wonderful to hear! I'm glad you're feeling happy.",
            "It's great that you're experiencing joy. What's bringing you happiness?",
        ],
        "gratitude": [
            "It's lovely that you feel grateful. Gratitude can be really healing.",
            "That's a beautiful perspective. What else are you thankful for?",
        ],
        "optimism": [
            "I love your positive outlook! What's making you feel hopeful?",
            "That's a great mindset to have. Optimism can really make a difference.",
        ],
    }

    conversations = []
    import random
    random.seed(42)

    for split in ["train"]:
        data = dataset[split]
        for row in data:
            text = row["text"].strip()
            labels = row["labels"]

            if not text or len(text) < 10:
                continue

            for label_id in labels:
                if label_id < len(emotion_labels):
                    emotion = emotion_labels[label_id]
                    if emotion in response_templates:
                        response = random.choice(response_templates[emotion])
                        conversations.append({
                            "input": text,
                            "response": response,
                            "emotion": emotion,
                            "source": "go_emotions",
                        })

    logger.info(f"Extracted {len(conversations)} conversation pairs from GoEmotions")
    return conversations


def load_counsel_chat() -> list[dict]:
    """
    Load and process the CounselChat dataset.
    Contains real counselor-patient Q&A pairs.
    """
    logger.info("Loading CounselChat dataset...")
    try:
        dataset = load_dataset("nbertagnolli/counsel-chat", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load CounselChat dataset: {e}")
        logger.info("Continuing without CounselChat data...")
        return []

    conversations = []
    data = dataset["train"]

    for row in data:
        question = row.get("questionTitle", "") or ""
        question_text = row.get("questionText", "") or ""
        answer = row.get("answerText", "") or ""

        # Combine question title and text
        full_question = question.strip()
        if question_text.strip():
            full_question = f"{full_question} {question_text.strip()}"

        full_question = full_question.strip()
        answer = answer.strip()

        if full_question and answer and len(full_question) > 10 and len(answer) > 20:
            # Truncate very long answers to keep them conversational
            if len(answer) > 500:
                answer = answer[:500].rsplit(" ", 1)[0] + "..."

            conversations.append({
                "input": full_question,
                "response": answer,
                "emotion": "support",
                "source": "counsel_chat",
            })

    logger.info(f"Extracted {len(conversations)} conversation pairs from CounselChat")
    return conversations


def format_for_dialogpt(conversations: list[dict]) -> list[dict]:
    """
    Format conversation pairs for DialoGPT fine-tuning.
    DialoGPT uses: input <eos> response <eos>
    """
    formatted = []
    for conv in conversations:
        input_text = conv["input"].strip()
        response_text = conv["response"].strip()

        if not input_text or not response_text:
            continue

        formatted.append({
            "text": f"{input_text}<|endoftext|>{response_text}<|endoftext|>",
            "input": input_text,
            "response": response_text,
            "emotion": conv.get("emotion", "neutral"),
            "source": conv.get("source", "unknown"),
        })

    return formatted


def create_dataset_splits(
    formatted_data: list[dict],
    config: Optional[DataConfig] = None,
) -> DatasetDict:
    """Create train/validation splits from the formatted data."""
    if config is None:
        config = DataConfig()

    import random
    random.seed(42)
    random.shuffle(formatted_data)

    split_idx = int(len(formatted_data) * config.train_split)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")

    return dataset_dict


def preprocess_pipeline(config: Optional[DataConfig] = None) -> DatasetDict:
    """
    Run the full preprocessing pipeline.
    Loads all datasets, processes them, and returns a unified DatasetDict.
    """
    if config is None:
        config = DataConfig()

    # Load all datasets
    all_conversations = []

    # 1. EmpatheticDialogues
    empathetic_convs = load_empathetic_dialogues()
    all_conversations.extend(empathetic_convs)

    # 2. GoEmotions
    go_emotions_convs = load_go_emotions()
    all_conversations.extend(go_emotions_convs)

    # 3. CounselChat
    counsel_convs = load_counsel_chat()
    all_conversations.extend(counsel_convs)

    logger.info(f"Total raw conversation pairs: {len(all_conversations)}")

    # Format for DialoGPT
    formatted = format_for_dialogpt(all_conversations)
    logger.info(f"Total formatted samples: {len(formatted)}")

    # Create splits
    dataset_dict = create_dataset_splits(formatted, config)

    # Save to disk
    os.makedirs(config.processed_data_dir, exist_ok=True)
    dataset_dict.save_to_disk(config.processed_data_dir)
    logger.info(f"Dataset saved to {config.processed_data_dir}")

    # Also save stats
    stats = {
        "total_samples": len(formatted),
        "train_samples": len(dataset_dict["train"]),
        "val_samples": len(dataset_dict["validation"]),
        "sources": {
            "empathetic_dialogues": len(empathetic_convs),
            "go_emotions": len(go_emotions_convs),
            "counsel_chat": len(counsel_convs),
        },
    }
    stats_path = os.path.join(config.processed_data_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")

    return dataset_dict


if __name__ == "__main__":
    config = DataConfig()
    dataset = preprocess_pipeline(config)
    print("\nDataset preview:")
    print(dataset["train"][0])
