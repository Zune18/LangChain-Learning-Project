"""
STEP 2 — Prompt Templates
==========================
Hard-coding prompts as strings is a mess. Prompt Templates let you:
  - Define structure once, reuse with different inputs
  - Inject variables cleanly with {placeholders}
  - Compose system + human messages (how real chat apps work)

Types used here:
  ChatPromptTemplate  — for chat models (system + human messages)
  PromptTemplate      — for simple single-string prompts
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# --- Simple Prompt (single string, one variable) ---
explain_prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms for a beginner. Be concise."
)


# --- Chat Prompt (system message + human message) ---
# System message sets the AI's persona/behavior.
# Human message is what the user sends.
tutor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding tutor. Always use simple analogies. Keep answers under 150 words."),
    ("human", "Explain {topic} with a real-world analogy.")
])


# --- Multi-variable Prompt ---
# You can inject multiple variables into one template.
custom_tone_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert who explains things in a {style} style."),
    ("human", "Explain {topic}.")
])


# --- Prompt with chat history (for memory — used in Step 3) ---
# MessagesPlaceholder is a slot where we'll inject past messages.
from langchain_core.prompts import MessagesPlaceholder

memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    MessagesPlaceholder(variable_name="history"),   # past messages injected here
    ("human", "{input}")                            # current user message
])