"""Module 08 — Prebuilt Components: create_react_agent, ToolNode & MessagesState

Demonstrates LangGraph's three main prebuilt abstractions:
  1. create_react_agent — one-line ReAct agent factory with auto tool-call loop
  2. ToolNode            — executes tool calls from messages, handles parallel calls
  3. MessagesState       — pre-wired TypedDict with add_messages reducer

Tools are defined with @tool from langchain_core and passed directly to the agent.
The calculator uses a safe AST-based arithmetic evaluator.

Real LangGraph pattern:
    from langgraph.prebuilt import create_react_agent, ToolNode
    agent = create_react_agent(model=get_llm(), tools=[calculator, web_search])
    result = agent.invoke({"messages": [("user", "What is 42 * 17?")]})
"""
import ast
import operator as _op
import re

from research_langgraph_demo.llm import get_llm

from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# -- Safe arithmetic evaluator (restricted AST) --

_ops = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        op_t = type(node.op)
        if op_t in _ops:
            return _ops[op_t](left, right)
        raise ValueError("Unsupported operator")
    if isinstance(node, ast.UnaryOp):
        op_t = type(node.op)
        if op_t in _ops:
            return _ops[op_t](_eval(node.operand))
        raise ValueError("Unsupported unary op")
    raise ValueError("Unsupported expression")


def safe_eval(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely via AST parsing."""
    try:
        tree = ast.parse(expr, mode="eval").body
        return str(_eval(tree))
    except Exception:
        return "error: could not evaluate expression"


# -- Tool definitions (only defined when langchain_core is available) --

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Supports +, -, *, /, ** operators."""
    # Strip non-math characters for robustness
    expr_clean = re.sub(r'[^0-9.\+\-\*\/\(\)\s]', '', expression).strip()
    if not expr_clean:
        return "error: no valid math expression found"
    return safe_eval(expr_clean)

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    return (
        f"Search results for '{query}': "
        f"Found 3 relevant articles about {query}. "
        f"Key finding: {query} is an active area of research with recent developments."
    )

@tool
def summarizer(text: str) -> str:
    """Summarize a piece of text into a concise one-sentence summary."""
    words = text.split()
    if len(words) <= 15:
        return text
    return " ".join(words[:15]) + "... (summarized)"


def run_demo() -> None:

    tools = [calculator, web_search, summarizer]
    llm = get_llm()

    # --- 1. create_react_agent: one-line ReAct agent with auto tool loop ---
    print("=== Prebuilt Components Demo ===\n")

    print("--- 1. create_react_agent ---")
    print("Building ReAct agent with tools:", [t.name for t in tools])
    agent = create_react_agent(model=llm, tools=tools)
    print(f"  Agent type: {type(agent).__name__}")
    print(f"  Graph nodes: {list(agent.get_graph().nodes.keys())}\n")

    # Query that should trigger the calculator tool
    print("--- 2. Agent invocation: math query ---")
    # result = agent.invoke(
    #     {"messages": [HumanMessage(content="What is 42 * 17 + 3?")]}
    # )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is (42 * 17) + (88 / 4) - 7?  Then search for 'Gemini 2.0' and summarize the result.")]}
    )

    for msg in result["messages"]:
        role = type(msg).__name__
        content = str(msg.content)[:150]
        print(f"  [{role}] {content}")
    print()

    # Query that should trigger web_search
    print("--- 3. Agent invocation: search query ---")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for recent advances in quantum computing")]}
    )
    for msg in result["messages"]:
        role = type(msg).__name__
        content = str(msg.content)[:150]
        print(f"  [{role}] {content}")
    print()

    # --- 4. ToolNode: standalone tool executor ---
    print("--- 4. ToolNode (standalone tool execution) ---")
    tool_node = ToolNode(tools)
    print(f"  ToolNode handles {len(tools)} tools: {[t.name for t in tools]}")
    print(f"  ToolNode type: {type(tool_node).__name__}")
    print()

    # --- 5. MessagesState ---
    print("--- 5. MessagesState ---")
    print(f"  MessagesState is a TypedDict with pre-wired add_messages reducer")
    print(f"  Keys: {list(MessagesState.__annotations__.keys())}")
    print(f"  Base classes: {[c.__name__ for c in MessagesState.__mro__ if c is not object]}")


if __name__ == "__main__":
    run_demo()