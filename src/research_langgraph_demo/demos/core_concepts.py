"""Module 01 — Core Concepts: StateGraph, Nodes, Edges & Routing

Demonstrates all five core LangGraph building blocks:
  1. StateGraph  — the central graph object defining computation topology
  2. Nodes       — Python functions that read/write shared state (TypedDict)
  3. Edges       — direct (unconditional) connections between nodes
  4. Conditional Edges — router functions that return the next node name
  5. START / END — special sentinel nodes to begin and terminate flow

The graph: START -> research -> evaluate --(conditional)--> refine -> evaluate
                                          `--(if good)----> END
"""
import re
from typing import TypedDict

from research_langgraph_demo.llm import llm_generate, HAS_GEMINI

try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


# -- State: shared TypedDict flowing through the graph --

class ResearchState(TypedDict):
    topic: str
    content: str
    score: int
    iterations: int


# -- Nodes: plain Python functions that read/write state --

def research(state: ResearchState) -> dict:
    """Generate a research paragraph on the topic."""
    content = llm_generate(
        f"Write one short paragraph about {state['topic']} for a research report."
    )
    return {"content": content, "iterations": state["iterations"] + 1}


def evaluate(state: ResearchState) -> dict:
    """Ask the LLM to score the content 1-10."""
    response = llm_generate(
        f"Rate the following research paragraph on a scale of 1-10 for quality, "
        f"accuracy, and completeness. Respond with ONLY a single integer.\n\n"
        f"{state['content']}"
    )
    try:
        score = int(response.strip())
    except ValueError:
        match = re.search(r"\d+", response)
        score = int(match.group()) if match else 5
    return {"score": score}


def refine(state: ResearchState) -> dict:
    """Rewrite the paragraph incorporating feedback."""
    content = llm_generate(
        f"Rewrite and improve this research paragraph about {state['topic']}. "
        f"Make it more detailed and accurate:\n\n{state['content']}"
    )
    return {"content": content}


# -- Conditional edge: router function returning next node name --

def route_after_eval(state: ResearchState) -> str:
    """If score >= 8 or max iterations reached, finish. Otherwise refine."""
    if state["score"] >= 8 or state["iterations"] >= 3:
        return END
    return "refine"


# -- Build and compile the graph --

def build_graph() -> "CompiledGraph":
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("research", research)
    graph.add_node("evaluate", evaluate)
    graph.add_node("refine", refine)

    # Add edges
    graph.add_edge(START, "research")        # START -> research
    graph.add_edge("research", "evaluate")   # research -> evaluate (unconditional)
    graph.add_edge("refine", "evaluate")     # refine -> evaluate (unconditional)

    # Conditional edge: evaluate -> END or refine
    graph.add_conditional_edges("evaluate", route_after_eval)

    return graph.compile()


def run_demo(topic: str = "AI safety") -> None:
    if not HAS_GEMINI:
        raise RuntimeError(
            "This demo requires a real LLM. Set GOOGLE_API_KEY and install langchain-google-genai."
        )
    if not HAS_LANGGRAPH:
        raise RuntimeError(
            "This demo requires langgraph. Install it with: pip install langgraph"
        )

    app = build_graph()

    # Invoke the graph with initial state
    initial_state = {
        "topic": topic,
        "content": "",
        "score": 0,
        "iterations": 0,
    }

    print(f"Running research graph for topic: '{topic}'\n")

    # Stream to see each step
    for step in app.stream(initial_state):
        for node_name, output in step.items():
            if node_name == "research":
                print(f"[research] Generated content (iter {output['iterations']})")
                print(f"  '{output['content'][:80]}...'\n")
            elif node_name == "evaluate":
                print(f"[evaluate] Score: {output['score']}\n")
            elif node_name == "refine":
                print(f"[refine] Improved content")
                print(f"  '{output['content'][:80]}...'\n")

    # Get final state
    final = app.invoke(initial_state)
    print("=" * 60)
    print(f"Final score: {final['score']}")
    print(f"Iterations:  {final['iterations']}")
    print(f"Final content:\n{final['content']}")


if __name__ == "__main__":
    run_demo()
