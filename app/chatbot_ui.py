"""
Gradio-based chatbot UI for the Mental Health Chatbot.

Features:
- Real-time conversation with the fine-tuned model
- Safety guardrails for crisis detection
- Conversation history management
- Disclaimer and helpline information
"""

import os
import sys
import logging

import gradio as gr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, InferenceConfig, SafetyConfig
from inference.generate import MentalHealthChatbot

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot_instance = None


def initialize_chatbot(use_adapter: bool = True) -> MentalHealthChatbot:
    """Initialize the chatbot model."""
    global chatbot_instance

    if chatbot_instance is not None:
        return chatbot_instance

    model_config = ModelConfig()
    inference_config = InferenceConfig()
    safety_config = SafetyConfig()

    chatbot_instance = MentalHealthChatbot(
        model_config=model_config,
        inference_config=inference_config,
        safety_config=safety_config,
        use_adapter=use_adapter,
    )
    chatbot_instance.load_model()
    return chatbot_instance


def respond(message: str, chat_history: list) -> tuple:
    """
    Generate a response and update the chat history.

    Args:
        message: User's input message
        chat_history: List of (user_msg, bot_msg) tuples

    Returns:
        Updated chat_history and empty string for input box
    """
    if not message.strip():
        return chat_history, ""

    bot = chatbot_instance
    if bot is None:
        return chat_history + [
            (message, "Error: Model not loaded. Please restart the application.")
        ], ""

    try:
        response = bot.generate_response(message.strip())
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response = (
            "I'm having trouble processing your message right now. "
            "Could you try rephrasing that? I'm here to listen."
        )

    chat_history = chat_history + [(message, response)]
    return chat_history, ""


def clear_chat() -> tuple:
    """Clear conversation history."""
    if chatbot_instance is not None:
        chatbot_instance.reset_conversation()
    return [], ""


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
    safety_config = SafetyConfig()

    with gr.Blocks(
        title="Mental Health Support Chatbot",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
        css="""
        .disclaimer {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            font-size: 14px;
        }
        .helpline {
            background-color: #d4edda;
            border: 1px solid #28a745;
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            font-size: 14px;
        }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 🧠 Mental Health Support Chatbot
            ### An AI-powered empathetic conversation partner

            *Powered by DialoGPT fine-tuned with QLoRA on mental health conversation data*
            """
        )

        # Disclaimer
        gr.Markdown(
            f'<div class="disclaimer">{safety_config.disclaimer}</div>'
        )

        # Chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=450,
            bubble_full_width=False,
            avatar_images=(None, "🤖"),
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Share how you're feeling today...",
                scale=4,
                lines=2,
            )
            send_btn = gr.Button("Send 💬", scale=1, variant="primary")

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        # Helpline information
        gr.Markdown(
            """
            <div class="helpline">

            ### 📞 Crisis Resources
            If you or someone you know is in crisis, please reach out:
            - **National Suicide Prevention Lifeline**: Call or text **988**
            - **Crisis Text Line**: Text **HOME** to **741741**
            - **SAMHSA Helpline**: **1-800-662-4357** (free, confidential, 24/7)
            - **International Crisis Lines**: [Find your local helpline](https://www.iasp.info/resources/Crisis_Centres/)

            </div>
            """
        )

        # Event handlers
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(clear_chat, [], [chatbot, msg])

        # Example prompts
        gr.Examples(
            examples=[
                "I've been feeling really anxious about work lately.",
                "I'm struggling with loneliness and don't know who to talk to.",
                "I had a panic attack today and I'm scared it will happen again.",
                "I feel overwhelmed with everything going on in my life.",
                "I lost someone close to me and I can't stop thinking about them.",
                "Some days I just don't feel motivated to do anything.",
            ],
            inputs=msg,
            label="💡 Try these conversation starters:",
        )

    return demo


def main():
    """Launch the chatbot application."""
    import argparse

    parser = argparse.ArgumentParser(description="Mental Health Chatbot UI")
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run without the QLoRA adapter (base DialoGPT only)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on",
    )
    args = parser.parse_args()

    # Initialize model
    logger.info("Initializing chatbot...")
    use_adapter = not args.no_adapter
    initialize_chatbot(use_adapter=use_adapter)
    logger.info("Chatbot initialized!")

    # Build and launch UI
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
