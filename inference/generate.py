"""
Inference pipeline for the Mental Health Chatbot.

Loads the base DialoGPT model with the QLoRA adapter and generates
empathetic responses to user messages.
"""

import os
import sys
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, InferenceConfig, SafetyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MentalHealthChatbot:
    """
    Mental Health Chatbot powered by DialoGPT + QLoRA.
    Generates empathetic responses with safety guardrails.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        use_adapter: bool = True,
    ):
        self.model_config = model_config or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.safety_config = safety_config or SafetyConfig()
        self.use_adapter = use_adapter
        self.conversation_history = []

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model with or without the QLoRA adapter."""
        logger.info(f"Loading tokenizer from {self.model_config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer_name,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading base model: {self.model_config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            device_map=self.model_config.device_map,
            torch_dtype=torch.float16,
        )

        # Load QLoRA adapter if available
        if self.use_adapter:
            adapter_path = self.inference_config.adapter_path
            if os.path.exists(adapter_path):
                logger.info(f"Loading QLoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                )
                logger.info("Adapter loaded successfully!")
            else:
                logger.warning(
                    f"Adapter not found at {adapter_path}. "
                    "Using base model without fine-tuning."
                )
                self.use_adapter = False

        self.model.eval()
        logger.info("Model loaded and ready for inference.")

    def check_crisis(self, user_input: str) -> Optional[str]:
        """
        Check if the user's message contains crisis indicators.
        Returns the crisis response if detected, None otherwise.
        """
        user_lower = user_input.lower()
        for keyword in self.safety_config.crisis_keywords:
            if keyword in user_lower:
                logger.warning(f"Crisis keyword detected: '{keyword}'")
                return self.safety_config.crisis_response
        return None

    def generate_response(
        self,
        user_input: str,
        max_history_turns: int = 5,
    ) -> str:
        """
        Generate an empathetic response to the user's input.

        Args:
            user_input: The user's message
            max_history_turns: Maximum number of previous turns to include

        Returns:
            The chatbot's response string
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Safety check first
        crisis_response = self.check_crisis(user_input)
        if crisis_response:
            return crisis_response

        # Build conversation context
        # DialoGPT format: turn1 <eos> turn2 <eos> ... current_input <eos>
        eos_token = self.tokenizer.eos_token
        context_turns = self.conversation_history[-max_history_turns:]

        # Encode conversation history
        input_ids = torch.tensor([], dtype=torch.long)

        for turn in context_turns:
            turn_ids = self.tokenizer.encode(
                turn + eos_token,
                return_tensors="pt",
            ).squeeze(0)
            input_ids = torch.cat([input_ids, turn_ids])

        # Add current user input
        new_input_ids = self.tokenizer.encode(
            user_input + eos_token,
            return_tensors="pt",
        ).squeeze(0)
        input_ids = torch.cat([input_ids, new_input_ids]).unsqueeze(0)

        # Truncate if too long
        max_length = self.model_config.max_length
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]

        # Move to device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.inference_config.max_new_tokens,
                temperature=self.inference_config.temperature,
                top_p=self.inference_config.top_p,
                top_k=self.inference_config.top_k,
                repetition_penalty=self.inference_config.repetition_penalty,
                do_sample=self.inference_config.do_sample,
                num_beams=self.inference_config.num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output_ids[:, input_ids.shape[1]:]
        response = self.tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        ).strip()

        # Fallback response if empty
        if not response:
            response = (
                "I hear you, and I appreciate you sharing that with me. "
                "Could you tell me a bit more about how you're feeling?"
            )

        # Update conversation history
        self.conversation_history.append(user_input)
        self.conversation_history.append(response)

        return response

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    def get_disclaimer(self) -> str:
        """Return the safety disclaimer."""
        return self.safety_config.disclaimer


def create_chatbot(
    use_adapter: bool = True,
    adapter_path: Optional[str] = None,
) -> MentalHealthChatbot:
    """
    Factory function to create and initialize a chatbot instance.

    Args:
        use_adapter: Whether to load the QLoRA adapter
        adapter_path: Custom path to the adapter (overrides config)

    Returns:
        Initialized MentalHealthChatbot instance
    """
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    safety_config = SafetyConfig()

    if adapter_path:
        inference_config.adapter_path = adapter_path

    chatbot = MentalHealthChatbot(
        model_config=model_config,
        inference_config=inference_config,
        safety_config=safety_config,
        use_adapter=use_adapter,
    )

    chatbot.load_model()
    return chatbot


if __name__ == "__main__":
    # Quick test
    print("Initializing Mental Health Chatbot...")
    bot = create_chatbot(use_adapter=False)

    print("\n" + bot.get_disclaimer())
    print("\n" + "=" * 60)
    print("Mental Health Chatbot (type 'quit' to exit, 'reset' to clear history)")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            print("Take care! Remember, it's okay to seek help when you need it.")
            break
        if user_input.lower() == "reset":
            bot.reset_conversation()
            print("Conversation reset.")
            continue
        if not user_input:
            continue

        response = bot.generate_response(user_input)
        print(f"\nBot: {response}")
