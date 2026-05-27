"""
STEP 2 — LCEL Chains (LangChain Expression Language)
======================================================
LCEL uses the pipe operator  |  to chain steps together.

  chain = prompt | model | parser

This reads left to right:
  1. prompt  → formats your input into a Message
  2. model   → sends the Message to the LLM, gets a response
  3. parser  → extracts the text from the response

Why LCEL?
  - Readable: you can SEE the data flow
  - Composable: chain any LangChain component with |
  - Streaming-ready: works with .stream() out of the box
  - Async-ready: .ainvoke() / .astream() just work

Output parsers:
  StrOutputParser       → plain string
  JsonOutputParser      → dict/list (expects JSON from model)
  CommaSeparatedParser  → list split on commas
"""

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.models.llm import get_llm
from app.prompts.templates import explain_prompt, tutor_prompt, custom_tone_prompt


# Shared components
llm = get_llm()
str_parser = StrOutputParser()   # just pulls .content off the AIMessage


# ── Chain 1: Explain a topic ──────────────────────────────────────────────────
# explain_prompt is a PromptTemplate (single string)
# It becomes a HumanMessage before hitting the model.
explain_chain = explain_prompt | llm | str_parser


# ── Chain 2: Tutor with system persona ───────────────────────────────────────
tutor_chain = tutor_prompt | llm | str_parser


# ── Chain 3: Custom tone ─────────────────────────────────────────────────────
tone_chain = custom_tone_prompt | llm | str_parser


# ── Chain 4: JSON output ──────────────────────────────────────────────────────
# Tell the model to respond as JSON, then parse it automatically.
json_prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond ONLY with valid JSON. No explanation. No markdown."),
    ("human", "Give me 3 key facts about {topic} as JSON: {{\"facts\": [\"...\", \"...\", \"...\"]}}"),
])
json_chain = json_prompt | llm | JsonOutputParser()


# ── Chain 5: Chained chains ───────────────────────────────────────────────────
# You can pipe chains together too.
# This explains a topic, then simplifies that explanation further.
from langchain_core.runnables import RunnableLambda

simplify_prompt = ChatPromptTemplate.from_messages([
    ("human", "Simplify this even further, use one sentence only:\n\n{text}")
])

# RunnableLambda wraps a plain Python function so it works inside LCEL pipes.
wrap_as_text = RunnableLambda(lambda text: {"text": text})

double_chain = explain_chain | wrap_as_text | simplify_prompt | llm | str_parser