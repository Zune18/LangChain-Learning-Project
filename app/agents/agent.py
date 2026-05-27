"""
STEP 5 — The Agent
===================
An Agent is an LLM that can:
  1. REASON  — decide what to do next
  2. ACT     — call a tool
  3. OBSERVE — read the tool result
  4. REPEAT  — until it has a final answer

This is the ReAct loop (Reason + Act).

create_react_agent builds this loop for you:
  - It creates the prompt with tool descriptions auto-injected
  - It handles the tool call → result → next step loop
  - It stops when the LLM decides it has a final answer

AgentExecutor wraps the agent and:
  - Runs the loop
  - Handles errors (handle_parsing_errors=True)
  - Limits runaway loops (max_iterations)
  - Can show you its thinking (verbose=True)

The agent gets ALL_TOOLS available. On every turn it:
  → reads the tool docstrings
  → decides which one (if any) to call
  → calls it, reads the result, reasons again
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from app.models.llm import get_llm
from app.tools.basic_tools import ALL_TOOLS


# The ReAct prompt — inlined so no LangChain Hub / LangSmith account is needed.
# This is the same prompt as hub.pull("hwchase17/react"), just local.
#
# {tools}          → auto-filled with tool names + docstrings
# {tool_names}     → auto-filled with just the tool names
# {input}          → the user's question
# {agent_scratchpad} → where Thought/Action/Observation steps accumulate
REACT_PROMPT = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")


def build_agent(verbose: bool = True) -> AgentExecutor:
    """
    Builds and returns a ready-to-use ReAct agent.

    verbose=True shows you the agent's thinking process:
      Thought: I need to search for this...
      Action: wikipedia_search
      Action Input: "neural networks"
      Observation: Neural networks are...
      Thought: I now have enough to answer...
      Final Answer: ...
    """
    llm = get_llm(temperature=0)   # temperature=0 for consistent tool-calling

    prompt = REACT_PROMPT

    # create_react_agent wires the prompt + model + tools together
    agent = create_react_agent(llm, ALL_TOOLS, prompt)

    # AgentExecutor is the runtime that actually runs the loop
    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,            # prints reasoning steps
        handle_parsing_errors=True, # don't crash on malformed tool calls
        max_iterations=8,           # safety limit — stops infinite loops
        return_intermediate_steps=False,
    )
    return executor