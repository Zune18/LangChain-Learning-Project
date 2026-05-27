"""
STEP 4 & 5 — Tools
===================
Tools give the LLM the ability to DO things, not just say things.

How it works:
  1. You define tools with @tool decorator
  2. You call model.bind_tools([...]) — this sends tool schemas to the LLM
  3. LLM sees the schemas and decides: "I should call calculator(5, 7)"
  4. LangChain executes the function and gives the result back to the LLM
  5. LLM uses the result to form its final answer

The @tool decorator:
  - Reads the function name → becomes the tool name
  - Reads the docstring    → the LLM reads this to know WHEN to use the tool
  - Reads the type hints   → auto-generates the JSON schema the LLM uses
  → ALWAYS write clear docstrings. The LLM literally reads them.
"""

from langchain.tools import tool
import math


# ── Tool 1: Calculator ────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this for any arithmetic, algebra, or math calculations.
    Input should be a valid Python math expression like '2 + 2' or 'sqrt(16)'.
    Examples: '5 * 7', '100 / 4', 'sqrt(144)', '2 ** 10'
    """
    try:
        # Allow math functions like sqrt, sin, cos, etc.
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, allowed)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ── Tool 2: Word Counter ──────────────────────────────────────────────────────
@tool
def word_counter(text: str) -> str:
    """
    Counts the number of words, characters, and sentences in a given text.
    Use this when the user asks about length, word count, or text statistics.
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    return (
        f"Words: {words} | Characters: {chars} | "
        f"Sentences: {max(sentences, 1)}"
    )


# ── Tool 3: Wikipedia Search ──────────────────────────────────────────────────
@tool
def wikipedia_search(query: str) -> str:
    """
    Searches Wikipedia and returns a summary of the topic.
    Use this to look up factual information, definitions, history,
    or background on any topic, person, place, concept, or event.
    """
    try:
        import wikipedia
        wikipedia.set_lang("en")
        result = wikipedia.summary(query, sentences=4, auto_suggest=True)
        return result
    except Exception as e:
        return f"Wikipedia search failed for '{query}': {e}"


# ── Tool 4: DuckDuckGo Web Search ─────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Searches the web using DuckDuckGo for recent or real-time information.
    Use this for current events, recent news, prices, or anything that
    might not be in the LLM's training data.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"No web results found for '{query}'"
        output = []
        for r in results:
            output.append(f"• {r['title']}\n  {r['body'][:200]}...")
        return "\n\n".join(output)
    except Exception as e:
        return f"Web search failed for '{query}': {e}"


# ── Tool 5: Save to File ──────────────────────────────────────────────────────
@tool
def save_to_file(content: str, filename: str = "output.txt") -> str:
    """
    Saves text content to a file in the data/ directory.
    Use this when the user wants to save, export, or write results to a file.
    """
    import os
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    with open(filepath, "w") as f:
        f.write(content)
    return f"Saved to data/{filename} ({len(content)} characters)"


# ── All tools as a list (imported by the agent) ───────────────────────────────
ALL_TOOLS = [calculator, word_counter, wikipedia_search, web_search, save_to_file]