# 🧠 Mental Health Support Chatbot

An end-to-end ML project that fine-tunes **DialoGPT** using **QLoRA** (Quantized Low-Rank Adaptation) on mental health conversation datasets to create an empathetic AI chatbot.

## Overview

This chatbot is designed to provide empathetic, supportive responses to users sharing their mental health concerns. It combines multiple datasets and uses parameter-efficient fine-tuning to create a conversational AI that understands emotional context.

> **⚠️ Disclaimer**: This is an AI chatbot for emotional support and is NOT a replacement for professional mental health care. If you are in crisis, please contact a crisis helpline immediately.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User (Gradio UI)                      │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│              Safety Guardrails                           │
│  (Crisis keyword detection → Helpline referral)         │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│           Inference Pipeline                             │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │  DialoGPT    │──│  QLoRA       │                     │
│  │  (Base)      │  │  Adapter     │                     │
│  └──────────────┘  └──────────────┘                     │
│  - Conversation history management                      │
│  - Temperature-controlled generation                    │
│  - Repetition penalty                                   │
└─────────────────────────────────────────────────────────┘
```

## Datasets

| Dataset | Source | Description | Size |
|---------|--------|-------------|------|
| [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues) | Facebook | Empathetic conversation pairs with emotion labels | ~25K conversations |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Google/Reddit | Reddit comments labeled with 27 emotions | ~58K comments |
| [CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) | CounselChat | Real counselor-patient Q&A pairs | ~3K pairs |

## Tech Stack

- **Base Model**: `microsoft/DialoGPT-medium`
- **Fine-tuning**: QLoRA (4-bit quantization + LoRA adapters)
- **Libraries**: transformers, peft, bitsandbytes, trl, accelerate
- **UI**: Gradio
- **Tracking**: Weights & Biases (optional)

## Project Structure

```
mental-health-chatbot/
├── config.py                 # All hyperparameters and settings
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # End-to-end pipeline runner
├── data/
│   ├── __init__.py
│   └── preprocess.py         # Dataset loading & preprocessing
├── training/
│   ├── __init__.py
│   └── train.py              # QLoRA fine-tuning script
├── inference/
│   ├── __init__.py
│   └── generate.py           # Inference pipeline & chatbot class
├── app/
│   ├── __init__.py
│   └── chatbot_ui.py         # Gradio chatbot interface
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd mental-health-chatbot

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Run everything: preprocess → train → launch chatbot
python run_pipeline.py --all
```

Or run each step individually:

### 3. Data Preprocessing

```bash
python -m data.preprocess
```

This will:
- Download EmpatheticDialogues, GoEmotions, and CounselChat datasets
- Clean and format conversation pairs
- Create train/validation splits
- Save processed data to `./data/processed/`

### 4. Fine-tuning with QLoRA

```bash
python -m training.train
```

This will:
- Load DialoGPT-medium with 4-bit quantization
- Apply LoRA adapters (rank=16, alpha=32)
- Fine-tune on the processed dataset
- Save the adapter weights to `./outputs/mental-health-chatbot/final_model/`

**Hardware Requirements**:
- Minimum: 1x GPU with 8GB VRAM (e.g., NVIDIA T4, RTX 3060)
- Recommended: 16GB+ VRAM for faster training
- Google Colab free tier (T4 GPU) works with QLoRA

### 5. Launch the Chatbot

```bash
# With fine-tuned adapter
python -m app.chatbot_ui

# Without adapter (base DialoGPT only)
python -m app.chatbot_ui --no-adapter

# Create a public link
python -m app.chatbot_ui --share

# Custom port
python -m app.chatbot_ui --port 8080
```

## Configuration

All hyperparameters are centralized in `config.py`:

### QLoRA Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha (scaling) |
| `lora_dropout` | 0.05 | LoRA dropout |
| `load_in_4bit` | True | 4-bit quantization |
| `bnb_4bit_quant_type` | nf4 | Quantization type |

### Training Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_epochs` | 3 | Number of training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `warmup_ratio` | 0.03 | Warmup proportion |

### Inference Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Sampling temperature |
| `top_p` | 0.9 | Nucleus sampling |
| `top_k` | 50 | Top-k sampling |
| `repetition_penalty` | 1.2 | Repetition penalty |
| `max_new_tokens` | 256 | Maximum response length |

## Safety Features

### Crisis Detection
The chatbot monitors user messages for crisis-related keywords and immediately provides helpline information when detected.

### Disclaimer
A clear disclaimer is displayed at all times, informing users that this is an AI tool and not a replacement for professional help.

### Crisis Resources
- **National Suicide Prevention Lifeline**: 988 (call or text)
- **Crisis Text Line**: Text HOME to 741741
- **SAMHSA Helpline**: 1-800-662-4357
- **International Crisis Lines**: [IASP](https://www.iasp.info/resources/Crisis_Centres/)

## Why QLoRA?

QLoRA (Quantized Low-Rank Adaptation) is ideal for this project because:

1. **Memory Efficient**: 4-bit quantization reduces the model's memory footprint by ~75%
2. **Parameter Efficient**: LoRA only trains ~0.5% of the total parameters
3. **Accessible**: Can fine-tune on a single consumer GPU (8GB VRAM)
4. **Quality**: Maintains model quality comparable to full fine-tuning
5. **Fast**: Training completes in hours rather than days

## Example Conversations

```
You: I've been feeling really anxious about work lately.
Bot: I understand that work anxiety can be really overwhelming. What specifically
     about work is causing you the most stress right now?

You: I feel like I can't keep up with all the deadlines and my boss keeps
     adding more tasks.
Bot: That sounds like a lot of pressure. It's completely valid to feel
     overwhelmed when the workload keeps increasing. Have you been able
     to communicate your concerns to your supervisor?
```

## Extending the Project

### Adding New Datasets
1. Add the dataset loader function in `data/preprocess.py`
2. Format outputs as `{"input": str, "response": str, "emotion": str, "source": str}`
3. Include in the `preprocess_pipeline()` function

### Changing the Base Model
Update `model_name` in `config.py`. Compatible models:
- `microsoft/DialoGPT-small` (lighter, faster)
- `microsoft/DialoGPT-large` (better quality, needs more VRAM)

### Using Weights & Biases
Set `report_to = "wandb"` in `TrainingConfig` and run:
```bash
wandb login
python -m training.train
```

## License

This project is for educational and research purposes. Please use responsibly and always prioritize professional mental health support for those in need.
