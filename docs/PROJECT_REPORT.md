# Mental Health Support Chatbot Using DialoGPT with QLoRA Fine-Tuning

## BTech Final Year Major Project Report

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Problem Statement](#4-problem-statement)
5. [Objectives](#5-objectives)
6. [System Architecture](#6-system-architecture)
7. [Methodology](#7-methodology)
8. [Datasets](#8-datasets)
9. [Model Architecture](#9-model-architecture)
10. [QLoRA Fine-Tuning](#10-qlora-fine-tuning)
11. [Implementation Details](#11-implementation-details)
12. [Safety and Ethical Considerations](#12-safety-and-ethical-considerations)
13. [Results and Evaluation](#13-results-and-evaluation)
14. [User Interface](#14-user-interface)
15. [Deployment](#15-deployment)
16. [Future Scope](#16-future-scope)
17. [Conclusion](#17-conclusion)
18. [References](#18-references)

---

## 1. Abstract

Mental health is an increasingly critical global concern, yet access to mental health support remains limited for many individuals due to cost, availability, and stigma. This project presents an AI-powered mental health support chatbot that provides empathetic, context-aware conversational assistance to users expressing emotional distress.

The system leverages **Microsoft's DialoGPT** (a GPT-2-based dialogue model) as the base architecture, fine-tuned using **QLoRA (Quantized Low-Rank Adaptation)** — a parameter-efficient fine-tuning technique that enables training large language models on consumer-grade hardware with minimal memory requirements.

The chatbot is trained on a combination of three curated datasets: **Facebook's EmpatheticDialogues** (~25,000 empathetic conversations), **Google's GoEmotions** (~58,000 emotion-labeled Reddit comments), and **CounselChat** (~3,000 real counselor-patient Q&A pairs). The system includes safety guardrails with crisis detection, helpline referral mechanisms, and a user-friendly Gradio-based web interface.

**Keywords**: Natural Language Processing, Mental Health, Chatbot, DialoGPT, QLoRA, LoRA, Fine-tuning, Empathetic Dialogue, Transformer, Parameter-Efficient Fine-Tuning

---

## 2. Introduction

### 2.1 Background

Mental health disorders affect approximately 1 in 8 people globally (WHO, 2022). Despite the prevalence, a significant treatment gap exists — particularly in developing nations where the ratio of mental health professionals to population is critically low. The COVID-19 pandemic further exacerbated this crisis, leading to a 25% increase in the prevalence of anxiety and depression worldwide.

Artificial Intelligence (AI), specifically Natural Language Processing (NLP), offers a promising avenue to bridge this gap. AI-powered chatbots can provide 24/7 accessible, non-judgmental, and scalable emotional support, serving as a first line of assistance for individuals who may not have immediate access to professional help.

### 2.2 Motivation

The motivation for this project stems from several key factors:

1. **Accessibility Gap**: Many people cannot afford or access mental health professionals
2. **Stigma Reduction**: Users may feel more comfortable sharing with an AI system initially
3. **24/7 Availability**: Unlike human counselors, AI chatbots can provide round-the-clock support
4. **Scalability**: A single system can serve thousands of users simultaneously
5. **Technological Advancement**: Recent breakthroughs in large language models (LLMs) and parameter-efficient fine-tuning make this feasible on limited hardware

### 2.3 Scope

This project encompasses:
- Data collection and preprocessing from multiple mental health conversation datasets
- Fine-tuning a pre-trained dialogue model using QLoRA
- Building a complete inference pipeline with safety features
- Developing an interactive web-based chatbot interface
- Evaluation of model performance using standard NLP metrics

---

## 3. Literature Review

### 3.1 Conversational AI for Mental Health

Several notable systems have been developed in this domain:

- **Woebot** (Fitzpatrick et al., 2017): A CBT-based chatbot that demonstrated significant reduction in depression symptoms over a 2-week trial
- **Wysa** (Inkster et al., 2018): An AI chatbot using evidence-based therapeutic techniques, shown to improve PHQ-9 depression scores
- **ELIZA** (Weizenbaum, 1966): The pioneering chatbot that used pattern matching to simulate a psychotherapist

### 3.2 Transformer-Based Language Models

- **GPT-2** (Radford et al., 2019): Introduced the autoregressive language model paradigm with 1.5B parameters
- **DialoGPT** (Zhang et al., 2020): Extended GPT-2 for multi-turn dialogue generation, trained on 147M conversation-like exchanges from Reddit
- **BlenderBot** (Roller et al., 2021): Facebook's open-domain chatbot emphasizing personality, knowledge, and empathy

### 3.3 Parameter-Efficient Fine-Tuning (PEFT)

- **LoRA** (Hu et al., 2021): Low-Rank Adaptation that freezes pre-trained weights and injects trainable rank decomposition matrices, reducing trainable parameters by 10,000x
- **QLoRA** (Dettmers et al., 2023): Extends LoRA with 4-bit NormalFloat quantization, enabling fine-tuning of 65B parameter models on a single 48GB GPU
- **Adapter Layers** (Houlsby et al., 2019): Inserting small trainable modules between transformer layers

### 3.4 Empathetic Dialogue Systems

- **EmpatheticDialogues** (Rashkin et al., 2019): Dataset and benchmark for training empathetic dialogue agents
- **MIME** (Majumder et al., 2020): Model that generates empathetic responses by mimicking emotions with appropriate polarity
- **CEM** (Sabour et al., 2022): Incorporates commonsense knowledge for improved empathetic response generation

---

## 4. Problem Statement

To design and implement an AI-powered mental health support chatbot that:

1. Understands user emotional states from conversational text
2. Generates empathetic, contextually appropriate responses
3. Maintains multi-turn conversation coherence
4. Includes safety mechanisms for crisis situations
5. Can be fine-tuned on limited hardware using parameter-efficient techniques
6. Provides an accessible web-based interface for user interaction

---

## 5. Objectives

### Primary Objectives
1. Fine-tune DialoGPT-medium using QLoRA on mental health conversation data
2. Build an end-to-end pipeline: data preprocessing, training, inference, and deployment
3. Implement safety guardrails including crisis detection and helpline referrals

### Secondary Objectives
4. Achieve lower perplexity on validation data compared to the base model
5. Generate responses that demonstrate emotional awareness and empathy
6. Create a user-friendly web interface with Gradio
7. Enable deployment on consumer-grade hardware (8GB VRAM GPU)

---

## 6. System Architecture

### 6.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                        USER INTERFACE                             |
|                    (Gradio Web Application)                       |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                      SAFETY LAYER                                |
|            Crisis Keyword Detection Module                       |
|    (Scans input for crisis indicators -> helpline referral)      |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                    INFERENCE ENGINE                               |
|  +-------------------+    +-------------------+                  |
|  |    DialoGPT       |    |    QLoRA          |                  |
|  |    Base Model      |--->    Adapter         |                  |
|  |    (Frozen)        |    |    (Fine-tuned)   |                  |
|  +-------------------+    +-------------------+                  |
|                                                                  |
|  - Conversation History Manager                                  |
|  - Token Generation (Temperature, Top-p, Top-k sampling)        |
|  - Repetition Penalty                                            |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                    TRAINING PIPELINE                              |
|  +------------------+  +------------------+  +------------------+|
|  | EmpatheticDialogs|  |   GoEmotions     |  |  CounselChat     ||
|  | (Facebook)       |  |   (Google/Reddit)|  |  (Counselors)    ||
|  +--------+---------+  +--------+---------+  +--------+---------+|
|           |                      |                     |         |
|           +----------------------+---------------------+         |
|                                  |                               |
|                    Data Preprocessing & Formatting                |
|                    QLoRA Fine-tuning (SFTTrainer)                |
+------------------------------------------------------------------+
```

### 6.2 Component Description

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base Model | DialoGPT-medium (355M params) | Pre-trained dialogue generation |
| Fine-tuning | QLoRA (PEFT + BitsAndBytes) | Parameter-efficient adaptation |
| Training | SFTTrainer (TRL library) | Supervised fine-tuning |
| Inference | Transformers + PEFT | Response generation |
| UI | Gradio | Web-based chat interface |
| Safety | Custom Python module | Crisis detection & referral |

---

## 7. Methodology

### 7.1 Overall Approach

The project follows a systematic methodology:

1. **Data Collection**: Aggregate data from three complementary sources
2. **Data Preprocessing**: Clean, format, and merge datasets into a unified format
3. **Model Selection**: Choose DialoGPT-medium as the base model for its conversational abilities
4. **QLoRA Configuration**: Set up 4-bit quantization and LoRA adapters
5. **Fine-tuning**: Train the model on processed mental health conversation data
6. **Evaluation**: Measure performance using perplexity and response quality metrics
7. **Deployment**: Build a Gradio-based web interface with safety features

### 7.2 Data Flow

```
Raw Datasets --> Preprocessing --> Formatted Pairs --> Tokenization --> Training
                                                                          |
User Input --> Safety Check --> Tokenize --> Model Inference --> Response Output
```

### 7.3 Training Strategy

- **Quantization**: 4-bit NormalFloat (NF4) quantization reduces model memory from ~1.4GB to ~350MB
- **LoRA Injection**: Trainable low-rank matrices added to attention and feed-forward layers
- **Optimizer**: Paged AdamW 32-bit (memory-efficient variant)
- **Learning Rate Schedule**: Cosine annealing with warmup
- **Gradient Accumulation**: Effective batch size of 16 (4 x 4 accumulation steps)

---

## 8. Datasets

### 8.1 EmpatheticDialogues (Facebook Research)

- **Source**: Rashkin et al. (2019), available on HuggingFace
- **Size**: ~25,000 conversations across 32 emotion categories
- **Format**: Multi-turn dialogues between a speaker describing an emotional situation and a listener providing empathetic responses
- **Emotions**: Happy, sad, angry, afraid, disgusted, surprised, and 26 more fine-grained emotions
- **Relevance**: Directly applicable for training empathetic response generation

### 8.2 GoEmotions (Google Research)

- **Source**: Demszky et al. (2020), extracted from Reddit comments
- **Size**: ~58,000 comments labeled with 27 emotion categories + neutral
- **Format**: Single-turn comments with multi-label emotion annotations
- **Processing**: Mapped emotion labels to empathetic response templates for mental-health-relevant emotions (sadness, fear, anger, nervousness, grief, etc.)
- **Relevance**: Provides diverse emotional expressions from real Reddit users

### 8.3 CounselChat

- **Source**: Bertagnolli (2020), collected from online counseling platforms
- **Size**: ~3,000 counselor-patient Q&A pairs
- **Format**: Patient questions paired with professional counselor responses
- **Relevance**: Provides high-quality, professionally-crafted therapeutic responses

### 8.4 Dataset Statistics

| Dataset | Raw Size | Processed Pairs | Source |
|---------|----------|----------------|--------|
| EmpatheticDialogues | ~25K conversations | Variable* | Facebook Research |
| GoEmotions | ~58K comments | Variable* | Google/Reddit |
| CounselChat | ~3K Q&A pairs | Variable* | CounselChat.com |
| **Total** | - | **Combined** | **3 sources** |

*Exact numbers depend on filtering criteria applied during preprocessing.

---

## 9. Model Architecture

### 9.1 DialoGPT (Base Model)

DialoGPT is built on the GPT-2 architecture (Radford et al., 2019):

- **Architecture**: Transformer decoder (autoregressive)
- **Variant Used**: DialoGPT-medium
- **Parameters**: ~355 million
- **Layers**: 24 transformer blocks
- **Hidden Size**: 1024
- **Attention Heads**: 16
- **Vocabulary**: 50,257 BPE tokens
- **Pre-training Data**: 147M conversation-like exchanges from Reddit (2005-2017)
- **Context Window**: 1024 tokens

### 9.2 Transformer Block Structure

Each transformer block contains:
1. **Multi-Head Self-Attention (c_attn)**: Computes query, key, value projections
2. **Attention Output Projection (c_proj)**: Projects concatenated attention heads
3. **Feed-Forward Network (c_fc)**: Two-layer MLP with GELU activation
4. **Layer Normalization**: Applied before each sub-layer (Pre-LN variant)

### 9.3 Why DialoGPT?

| Factor | Justification |
|--------|---------------|
| Pre-trained for dialogue | Already understands conversational patterns |
| Moderate size (355M) | Feasible for QLoRA on consumer GPUs |
| Open-source | Freely available on HuggingFace |
| GPT-2 backbone | Well-understood architecture with extensive tooling |
| Reddit pre-training | Exposure to diverse emotional expressions |

---

## 10. QLoRA Fine-Tuning

### 10.1 What is QLoRA?

QLoRA (Quantized Low-Rank Adaptation) combines two techniques:

1. **Quantization**: Reduces the precision of model weights from 32-bit floating point to 4-bit integers, reducing memory by ~75%
2. **LoRA**: Instead of updating all model weights, injects small trainable low-rank matrices into specific layers

### 10.2 Mathematical Foundation

**LoRA Decomposition:**

For a pre-trained weight matrix W_0 in R^(d x k), LoRA constrains the update:

```
W = W_0 + Delta_W = W_0 + B * A
```

Where:
- B in R^(d x r) and A in R^(r x k)
- r << min(d, k) is the rank (r = 16 in our case)
- Only B and A are trainable (W_0 is frozen)
- At inference: output = W_0 * x + (B * A) * x

**NF4 Quantization:**

NormalFloat 4-bit uses an information-theoretically optimal data type for normally distributed weights:
- Divides the normal distribution into 2^4 = 16 quantization bins
- Each bin has equal probability mass
- Results in lower quantization error compared to uniform 4-bit

### 10.3 QLoRA Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 16 | Rank of the low-rank decomposition |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.05 | Dropout applied to LoRA layers |
| `target_modules` | c_attn, c_proj, c_fc | Layers where LoRA is applied |
| `load_in_4bit` | True | Enable 4-bit quantization |
| `bnb_4bit_quant_type` | nf4 | NormalFloat 4-bit |
| `bnb_4bit_use_double_quant` | True | Nested quantization for extra savings |

### 10.4 Parameter Efficiency

| Metric | Value |
|--------|-------|
| Total Parameters | ~355M |
| Trainable Parameters (LoRA) | ~1.8M |
| Percentage Trainable | ~0.5% |
| Base Model Memory (FP32) | ~1.4 GB |
| Quantized Model Memory (NF4) | ~350 MB |
| Training Memory (with QLoRA) | ~4-6 GB |

### 10.5 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Standard for fine-tuning; avoids overfitting |
| Batch Size | 4 | Memory-efficient for T4 GPU |
| Gradient Accumulation | 4 | Effective batch size = 16 |
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| LR Scheduler | Cosine | Smooth decay for stable convergence |
| Warmup Ratio | 0.03 | Gradual warmup to avoid early instability |
| Optimizer | Paged AdamW 32-bit | Memory-efficient variant |
| Max Gradient Norm | 0.3 | Gradient clipping for stability |
| Weight Decay | 0.01 | Regularization |
| FP16 | True | Mixed precision training |
| Max Sequence Length | 512 | Covers most conversation turns |

---

## 11. Implementation Details

### 11.1 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.0+ |
| Model Hub | HuggingFace Transformers | 4.36+ |
| PEFT | HuggingFace PEFT | 0.7+ |
| Quantization | BitsAndBytes | 0.41+ |
| Training | TRL (Transformer Reinforcement Learning) | 0.7+ |
| Distributed | Accelerate | 0.25+ |
| UI | Gradio | 4.0+ |
| Data | HuggingFace Datasets | 2.16+ |
| Visualization | Matplotlib, Pandas | Latest |
| Experiment Tracking | Weights & Biases (optional) | 0.16+ |

### 11.2 Project Structure

```
mental-health-chatbot/
├── config.py                    # Centralized configuration (dataclasses)
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # End-to-end pipeline CLI runner
├── Mental_Health_Chatbot_Colab.ipynb  # Google Colab notebook
├── data/
│   ├── __init__.py
│   └── preprocess.py            # Dataset loading & preprocessing
├── training/
│   ├── __init__.py
│   └── train.py                 # QLoRA fine-tuning script
├── inference/
│   ├── __init__.py
│   └── generate.py              # Inference pipeline & chatbot class
├── app/
│   ├── __init__.py
│   └── chatbot_ui.py            # Gradio chatbot interface
└── docs/
    └── PROJECT_REPORT.md        # This document
```

### 11.3 Key Implementation Modules

#### 11.3.1 Data Preprocessing (`data/preprocess.py`)
- Loads three datasets from HuggingFace Hub
- Extracts conversation pairs (input -> response)
- Formats data as `input<|endoftext|>response<|endoftext|>` for DialoGPT
- Creates 90/10 train/validation splits
- Saves processed data and statistics to disk

#### 11.3.2 Training (`training/train.py`)
- Configures 4-bit quantization using BitsAndBytesConfig
- Loads DialoGPT-medium with quantization
- Applies LoRA adapters to attention and feed-forward layers
- Uses SFTTrainer for supervised fine-tuning
- Saves adapter weights (not full model) for efficient storage

#### 11.3.3 Inference (`inference/generate.py`)
- Loads base model + QLoRA adapter
- Manages multi-turn conversation history
- Implements crisis keyword detection before response generation
- Uses configurable generation parameters (temperature, top-p, top-k)
- Provides fallback responses for edge cases

#### 11.3.4 Web Interface (`app/chatbot_ui.py`)
- Gradio Blocks-based responsive UI
- Real-time conversation with the model
- Safety disclaimer displayed prominently
- Crisis helpline information always visible
- Example conversation starters for user guidance

---

## 12. Safety and Ethical Considerations

### 12.1 Crisis Detection

The system implements a keyword-based crisis detection module that scans every user message for indicators of:
- Suicidal ideation (e.g., "kill myself", "end my life", "want to die")
- Self-harm (e.g., "cutting myself", "hurt myself")
- Hopelessness (e.g., "no reason to live", "better off dead")

When crisis keywords are detected, the system immediately:
1. Bypasses the AI response generation
2. Provides crisis helpline numbers (988, Crisis Text Line)
3. Displays a compassionate, pre-written safety message

### 12.2 Disclaimer

A clear disclaimer is displayed at all times, stating:
- The chatbot is an AI tool, not a licensed professional
- It should not replace professional mental health care
- Users in crisis should contact emergency services

### 12.3 Ethical Guidelines

1. **Privacy**: No user conversations are stored or logged beyond the session
2. **Non-diagnosis**: The system explicitly avoids making diagnoses
3. **Referral**: Always points users to professional resources
4. **Transparency**: Clearly identifies itself as an AI
5. **Harm Prevention**: Crisis detection takes priority over all other functions

### 12.4 Limitations Acknowledged

- The chatbot may generate inappropriate or insensitive responses
- Keyword-based crisis detection has false positive/negative risks
- The system cannot replace professional mental health care
- Cultural sensitivity may be limited by training data biases

---

## 13. Results and Evaluation

### 13.1 Evaluation Metrics

The model is evaluated using the following metrics:

#### 13.1.1 Perplexity
Perplexity measures how well the model predicts the next token. Lower perplexity indicates better language modeling:

```
PPL = exp(-(1/N) * sum(log P(w_i | w_1, ..., w_{i-1})))
```

#### 13.1.2 Response Quality Metrics
- **Average Response Length**: Measures verbosity/conciseness
- **Response Diversity**: Unique n-grams in generated responses
- **Emotional Coherence**: Whether responses match the emotional context

#### 13.1.3 Safety Metrics
- **Crisis Detection Recall**: Percentage of crisis messages correctly identified
- **False Positive Rate**: Non-crisis messages incorrectly flagged

### 13.2 Expected Results

After QLoRA fine-tuning:
- Perplexity should decrease compared to the base DialoGPT model
- Responses should demonstrate awareness of emotional context
- Crisis detection should achieve high recall (>95%) for known keywords
- Multi-turn conversations should maintain coherence

### 13.3 Qualitative Examples

**Example 1: Anxiety**
```
User: I've been feeling really anxious about work lately.
Bot:  I understand that work anxiety can be really overwhelming.
      What specifically about work is causing you the most stress right now?
```

**Example 2: Loneliness**
```
User: I'm struggling with loneliness and don't know who to talk to.
Bot:  Loneliness is such a difficult feeling. I'm glad you're
      reaching out here. Would you like to talk about what's been
      making you feel isolated?
```

**Example 3: Crisis Detection**
```
User: I feel like I want to end my life.
Bot:  I'm really concerned about what you're sharing. Please reach
      out to a crisis helpline immediately:
      - National Suicide Prevention Lifeline: 988
      - Crisis Text Line: Text HOME to 741741
```

---

## 14. User Interface

### 14.1 Design Principles
- **Simplicity**: Clean, uncluttered interface that doesn't overwhelm users in distress
- **Safety First**: Disclaimer and helpline information always visible
- **Accessibility**: Works on any device with a web browser
- **Privacy**: No login required, no data stored

### 14.2 Interface Components
1. **Header**: Project title and description
2. **Disclaimer Banner**: Yellow-highlighted safety disclaimer
3. **Chat Window**: Scrollable conversation history with user/bot message bubbles
4. **Input Area**: Text input with send button
5. **Controls**: Clear chat button
6. **Crisis Resources**: Green-highlighted section with helpline numbers
7. **Example Prompts**: Suggested conversation starters

### 14.3 Technology
- Built with Gradio Blocks API for full customization
- Custom CSS for disclaimer and helpline styling
- Responsive design works on mobile and desktop
- Supports public URL sharing (via Gradio's share feature)

---

## 15. Deployment

### 15.1 Google Colab Deployment
A ready-to-use Google Colab notebook is provided (`Mental_Health_Chatbot_Colab.ipynb`) that:
1. Clones the repository
2. Installs all dependencies
3. Runs data preprocessing
4. Executes QLoRA fine-tuning
5. Tests the chatbot
6. Launches a public Gradio URL
7. Optionally saves the model to Google Drive

**Requirements**: Free Google Colab account with T4 GPU runtime

### 15.2 Local Deployment
```bash
pip install -r requirements.txt
python run_pipeline.py --all
```

### 15.3 Hardware Requirements

| Setup | GPU | VRAM | Training Time |
|-------|-----|------|---------------|
| Minimum | NVIDIA T4 | 8 GB | ~3 hours |
| Recommended | NVIDIA RTX 3060 | 12 GB | ~2 hours |
| Optimal | NVIDIA A100 | 40 GB | ~30 minutes |
| Cloud | Google Colab (T4) | 15 GB | ~2 hours |

---

## 16. Future Scope

### 16.1 Short-Term Improvements
1. **Emotion Classification Head**: Add an explicit emotion classification module to better understand user emotional state
2. **Retrieval-Augmented Generation (RAG)**: Integrate a knowledge base of mental health resources
3. **Multi-language Support**: Extend to support conversations in Hindi, Spanish, and other languages
4. **Improved Crisis Detection**: Replace keyword matching with a trained classifier to reduce false positives

### 16.2 Medium-Term Extensions
5. **Voice Integration**: Add speech-to-text and text-to-speech for voice-based interaction
6. **Sentiment Tracking**: Track user sentiment across sessions to detect trends
7. **Professional Dashboard**: Create a monitoring interface for mental health professionals
8. **A/B Testing Framework**: Systematically compare different model configurations

### 16.3 Long-Term Vision
9. **Reinforcement Learning from Human Feedback (RLHF)**: Train using feedback from licensed therapists
10. **Personalization**: Adapt conversation style based on user preferences
11. **Multi-modal Input**: Support image and audio inputs for richer expression
12. **Clinical Validation**: Conduct clinical trials to measure therapeutic efficacy

---

## 17. Conclusion

This project successfully demonstrates the development of an end-to-end mental health support chatbot using modern NLP techniques. By leveraging DialoGPT as a conversational foundation and QLoRA for parameter-efficient fine-tuning, we created a system that:

1. **Generates empathetic responses** to users expressing emotional distress
2. **Runs on consumer hardware** thanks to 4-bit quantization and LoRA adapters
3. **Prioritizes user safety** through crisis detection and helpline referrals
4. **Provides an accessible interface** via a web-based Gradio application
5. **Is easily reproducible** with a Google Colab notebook and comprehensive documentation

The project highlights the potential of AI in supporting mental health while acknowledging the critical importance of professional care. The combination of QLoRA's efficiency with DialoGPT's conversational capabilities demonstrates that meaningful NLP applications can be built without requiring massive computational resources.

While the system has limitations — including the simplicity of keyword-based crisis detection and potential biases in training data — it serves as a strong foundation for further development and research in AI-assisted mental health support.

---

## 18. References

1. Fitzpatrick, K. K., Darcy, A., & Vierhile, M. (2017). Delivering cognitive behavior therapy to young adults with symptoms of depression via a fully automated conversational agent (Woebot). *JMIR Mental Health*, 4(2), e19.

2. Inkster, B., Sarda, S., & Subramanian, V. (2018). An empathy-driven, conversational artificial intelligence agent (Wysa) for digital mental well-being. *JMIR mHealth and uHealth*, 6(11), e12106.

3. Weizenbaum, J. (1966). ELIZA — A computer program for the study of natural language communication between man and machine. *Communications of the ACM*, 9(1), 36-45.

4. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI blog*, 1(8), 9.

5. Zhang, Y., Sun, S., Galley, M., Chen, Y. C., Brockett, C., Gao, X., ... & Dolan, B. (2020). DialoGPT: Large-scale generative pre-training for conversational response generation. *ACL 2020*.

6. Roller, S., Dinan, E., Goyal, N., Ju, D., Williamson, M., Liu, Y., ... & Weston, J. (2021). Recipes for building an open-domain chatbot. *EACL 2021*.

7. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

8. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized language models. *NeurIPS 2023*.

9. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. *ICML 2019*.

10. Rashkin, H., Smith, E. M., Li, M., & Boureau, Y. L. (2019). Towards empathetic open-domain conversation models: A new benchmark and dataset. *ACL 2019*.

11. Majumder, N., Hong, P., Peng, S., Lu, J., Ghosal, D., Gelbukh, A., ... & Poria, S. (2020). MIME: MIMicking emotions for empathetic response generation. *EMNLP 2020*.

12. Sabour, S., Zheng, C., & Huang, M. (2022). CEM: Commonsense-aware empathetic response generation. *AAAI 2022*.

13. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *ACL 2020*.

14. World Health Organization. (2022). World mental health report: Transforming mental health for all.

15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.

---

*This project was developed as a BTech Final Year Major Project.*
