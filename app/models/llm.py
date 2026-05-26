"""
STEP 1 — LLM Setup
==================
This file is the single source of truth for your LLM.
You import get_llm() everywhere instead of re-creating the model.

Key concept: ChatOpenAI is LangChain's model abstraction.
It works with OpenAI, OpenRouter, Azure, etc. — same interface,
just different base_url and api_key. This is the power of LangChain.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Returns a configured LLM instance.

    temperature controls creativity:
      0.0 = deterministic / factual
      0.7 = balanced (default)
      1.0 = creative / random

    We use OpenRouter as the backend, which lets you swap
    the model string to use GPT-4.1, Claude, Gemini, etc.
    without changing any other code.
    """
    return ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1000 # max tokens the model is allowed to generate in response
    )

# IGNORE
# from langchain_core.messages import HumanMessage, SystemMessage

# # Get the LLM instance (adjusting temperature if you want)
# llm = get_llm(temperature=0.2)  # Lower temperature for more factual answers

# # Structure your conversation
# messages = [
#     SystemMessage(content="You are a sarcastic comedian."),
#     HumanMessage(content="Explain quantum physics in one sentence."),
# ]

# # Invoke the model
# response = llm.invoke(messages)

# # Print the result
# print(response.content)