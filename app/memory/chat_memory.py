"""
STEP 3 — Chat Memory
=====================
LLMs are stateless — they remember nothing between calls.
Memory is just us manually passing conversation history back
into the prompt on every turn.

LangChain gives us helper classes to manage this cleanly.

Concepts:
  ChatMessageHistory   — simple in-memory list of messages
  RunnableWithMessageHistory — wraps any chain, handles injecting
                               history into the prompt automatically

Message types:
  HumanMessage  — what the user said
  AIMessage     — what the model replied
  SystemMessage — background instructions (not shown to user)

session_id:
  History is stored per session_id so you can run multiple
  independent conversations from the same code.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from app.models.llm import get_llm
from app.prompts.templates import memory_prompt


# In-memory store: { session_id: ChatMessageHistory }
# In a real app this would be Redis / PostgreSQL / DynamoDB.
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the message history for a session.
    Creates a new empty history if this session_id is new.
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def build_memory_chain():
    """
    Builds a chain that automatically remembers conversation history.

    How it works:
      1. User sends {input}
      2. RunnableWithMessageHistory fetches history for session_id
      3. History is injected into memory_prompt's {history} slot
      4. Full prompt (history + current input) is sent to LLM
      5. AI reply is appended to history for next turn
    """
    llm = get_llm(temperature=0.5)

    base_chain = memory_prompt | llm | StrOutputParser()

    chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",       # which key is the current user message
        history_messages_key="history",   # which key gets the past messages injected
    )
    return chain_with_memory


def clear_session(session_id: str):
    """Wipe history for a session (like starting a new chat)."""
    if session_id in _store:
        del _store[session_id]


def get_history(session_id: str) -> list:
    """Return raw message history for inspection/debugging."""
    history = get_session_history(session_id)
    return history.messages