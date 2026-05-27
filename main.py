"""
main.py — LangChain Learning Project
======================================
Run this file to step through each LangChain concept interactively.

  python main.py

Each step is self-contained. Complete them in order — each one
builds on the concepts from the last.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str):
    print("\n" + "═" * 55)
    print(f"  {title}")
    print("═" * 55)

def section(title: str):
    print(f"\n── {title} " + "─" * (45 - len(title)))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Simple LLM Call
# ─────────────────────────────────────────────────────────────────────────────

def step1_simple_llm():
    banner("STEP 1 — Simple LLM Call")

    print("""
What you'll learn:
  • How to create a ChatOpenAI model
  • The difference between invoke(), stream(), batch()
  • What an AIMessage looks like
    """)

    from app.models.llm import get_llm
    llm = get_llm(temperature=0.7)

    # ── invoke(): single call, waits for full response ──────────────────────
    section("invoke() — waits for full response")
    response = llm.invoke("What is LangChain in one sentence?")

    print(f"\nFull AIMessage object:\n  {response}")
    print(f"\nJust the text (.content):\n  {response.content}")

    # ── stream(): prints tokens as they arrive ───────────────────────────────
    section("stream() — tokens arrive in real time")
    print("\nStreaming response: ", end="", flush=True)
    for chunk in llm.stream("Name 3 uses of LLMs. Be brief."):
        print(chunk.content, end="", flush=True)
    print()

    # ── batch(): multiple prompts in one call ────────────────────────────────
    section("batch() — multiple prompts at once")
    responses = llm.batch([
        "What is Python in one word?",
        "What is SQL in one word?",
        "What is Docker in one word?",
    ])
    for r in responses:
        print(f"  → {r.content}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Prompt Templates + LCEL Chains
# ─────────────────────────────────────────────────────────────────────────────

def step2_chains():
    banner("STEP 2 — Prompt Templates + LCEL Chains")

    print("""
What you'll learn:
  • ChatPromptTemplate  → structures system + human messages
  • StrOutputParser     → extracts plain text from AIMessage
  • The pipe |          → chains components left-to-right
  • JsonOutputParser    → auto-parses JSON responses
  • Chained chains      → piping one chain's output into another
    """)

    from app.chains.basic_chain import (
        explain_chain, tutor_chain, tone_chain,
        json_chain, double_chain
    )

    # ── Chain 1 ──────────────────────────────────────────────────────────────
    section("explain_chain  (simple PromptTemplate → LLM → StrParser)")
    result = explain_chain.invoke({"topic": "embeddings"})
    print(result)

    # ── Chain 2 ──────────────────────────────────────────────────────────────
    section("tutor_chain  (ChatPromptTemplate with system message)")
    result = tutor_chain.invoke({"topic": "tokens"})
    print(result)

    # ── Chain 3 ──────────────────────────────────────────────────────────────
    section("tone_chain  (two variables: style + topic)")
    result = tone_chain.invoke({"style": "pirate", "topic": "APIs"})
    print(result)

    # ── Chain 4 ──────────────────────────────────────────────────────────────
    section("json_chain  (JsonOutputParser → returns a dict)")
    result = json_chain.invoke({"topic": "vector databases"})
    print(f"Type: {type(result)}")  # dict, not string!
    for i, fact in enumerate(result.get("facts", []), 1):
        print(f"  {i}. {fact}")

    # ── Chain 5 ──────────────────────────────────────────────────────────────
    section("double_chain  (chain piped into another chain)")
    result = double_chain.invoke({"topic": "neural networks"})
    print(f"One-sentence simplification:\n  {result}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Chat Memory
# ─────────────────────────────────────────────────────────────────────────────

def step3_memory():
    banner("STEP 3 — Chat Memory")

    print("""
What you'll learn:
  • Why LLMs are stateless (they forget everything)
  • ChatMessageHistory  → stores messages per session
  • RunnableWithMessageHistory → auto-injects history into prompts
  • session_id → run multiple independent chats from same code
    """)

    from app.memory.chat_memory import build_memory_chain, get_history, clear_session

    chain = build_memory_chain()
    session = "demo-session"

    def chat(message: str) -> str:
        return chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": session}}
        )

    # ── Multi-turn conversation ───────────────────────────────────────────────
    section("Multi-turn conversation (AI remembers context)")

    turns = [
        "My name is Alex and I'm learning LangChain.",
        "What's my name? And what am I learning?",    # tests memory
        "Give me one tip for what I'm learning.",      # tests context
    ]

    for msg in turns:
        print(f"\n🧑 You: {msg}")
        reply = chat(msg)
        print(f"🤖 AI : {reply}")

    # ── Inspect stored history ────────────────────────────────────────────────
    section("Inspecting stored message history")
    history = get_history(session)
    print(f"\n{len(history)} messages stored:")
    for msg in history:
        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
        print(f"  [{role}] {msg.content[:80]}...")

    # ── Multiple sessions ─────────────────────────────────────────────────────
    section("Multiple independent sessions")
    chat_b = lambda msg: chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": "session-B"}}
    )
    chat("Earlier I said my name was Alex.")
    r = chat_b("What's my name?")
    print(f"\nSession B doesn't know Alex:\n  {r}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Tool Calling
# ─────────────────────────────────────────────────────────────────────────────

def step4_tools():
    banner("STEP 4 — Tool Calling")

    print("""
What you'll learn:
  • @tool decorator → turns a function into an LLM-callable tool
  • model.bind_tools() → sends tool schemas to the LLM
  • AIMessage.tool_calls → the LLM's decision to call a tool
  • ToolMessage → how you return results back to the LLM
  • The LLM decides WHEN and WHETHER to use a tool
    """)

    from langchain_core.messages import ToolMessage
    from app.models.llm import get_llm
    from app.tools.basic_tools import calculator, word_counter, ALL_TOOLS

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def run_one_tool_call(prompt: str):
        """Manually demonstrates the tool-calling cycle."""
        print(f"\n🧑 User: {prompt}")

        response = llm_with_tools.invoke(prompt)

        if not response.tool_calls:
            print(f"🤖 AI (no tool needed): {response.content}")
            return

        for call in response.tool_calls:
            print(f"\n🔧 Tool call decided by LLM:")
            print(f"   Name : {call['name']}")
            print(f"   Args : {call['args']}")

            # Find the tool and execute it
            tool_map = {t.name: t for t in ALL_TOOLS}
            tool_result = tool_map[call["name"]].invoke(call["args"])
            print(f"   Result: {tool_result}")

            # Feed the result back to get a final natural-language answer
            messages = [
                {"role": "user", "content": prompt},
                response,
                ToolMessage(content=str(tool_result), tool_call_id=call["id"]),
            ]
            final = llm_with_tools.invoke(messages)
            print(f"\n🤖 Final answer: {final.content}")

    section("Calculator tool")
    run_one_tool_call("What is 144 divided by 12, then multiply by 7?")

    section("Word counter tool")
    run_one_tool_call("Count the words in: 'LangChain makes building LLM apps much easier'")

    section("No tool needed")
    run_one_tool_call("What is the capital of France?")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Full Agent (ReAct Loop)
# ─────────────────────────────────────────────────────────────────────────────

def step5_agent():
    banner("STEP 5 — Full ReAct Agent")

    print("""
What you'll learn:
  • ReAct = Reason + Act loop
  • create_react_agent → builds the agent
  • AgentExecutor → runs the Thought/Action/Observation loop
  • The agent picks tools on its own, chains multiple calls if needed
  • verbose=True shows you EVERY reasoning step

Tools available to the agent:
  • calculator       — math expressions
  • word_counter     — text statistics
  • wikipedia_search — factual lookups
  • web_search       — real-time web info
  • save_to_file     — persist results
    """)

    from app.agents.agent import build_agent

    agent = build_agent(verbose=True)   # verbose shows Thought/Action/Observation

    queries = [
        "What is 15% of 2400? Show your work.",
        "Search Wikipedia for 'transformer neural network' and summarize it in 2 sentences.",
        "Search for 'transformer neural network' on Wikipedia, count how many words are in the summary, then save the summary to transformers.txt",
    ]

    for i, query in enumerate(queries, 1):
        section(f"Query {i}")
        print(f"\n🧑 User: {query}\n")
        try:
            result = agent.invoke({"input": query})
            print(f"\n✅ Final Answer:\n{result['output']}")
        except Exception as e:
            print(f"⚠ Agent error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────

def interactive_agent():
    """Free-form chat with the agent — try anything!"""
    banner("INTERACTIVE AGENT — Ask me anything")
    print("Type 'quit' to exit.\n")
    print("Try: 'What is sqrt(256)?'")
    print("     'Search Wikipedia for LangChain'")
    print("     'What is 12 * 15 and save the answer to math.txt'\n")

    from app.agents.agent import build_agent
    agent = build_agent(verbose=False)  # cleaner output for interactive use

    while True:
        query = input("🧑 You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        try:
            result = agent.invoke({"input": query})
            print(f"🤖 Agent: {result['output']}\n")
        except Exception as e:
            print(f"⚠ Error: {e}\n")


STEPS = {
    "1": ("Simple LLM Call",              step1_simple_llm),
    "2": ("Prompt Templates + LCEL",       step2_chains),
    "3": ("Chat Memory",                   step3_memory),
    "4": ("Tool Calling",                  step4_tools),
    "5": ("Full ReAct Agent",              step5_agent),
    "6": ("Interactive Agent (free chat)", interactive_agent),
}


def main():
    print("""
╔═══════════════════════════════════════════════════╗
║        LangChain Learning Project                 ║
║        Python · OpenRouter · Step by Step         ║
╚═══════════════════════════════════════════════════╝

Steps:
""")
    for k, (label, _) in STEPS.items():
        print(f"  {k}. {label}")

    print("\n  all  — run all steps in order")
    print("  q    — quit\n")

    choice = input("Run step: ").strip().lower()

    if choice == "q":
        return
    elif choice == "all":
        for k, (label, fn) in STEPS.items():
            if k == "6":           # skip interactive in 'all' mode
                continue
            fn()
            input("\n  ↵  Press Enter for next step...")
    elif choice in STEPS:
        STEPS[choice][1]()
    else:
        print("Unknown choice.")


if __name__ == "__main__":
    main()