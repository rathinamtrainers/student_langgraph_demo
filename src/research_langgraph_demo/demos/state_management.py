"""Module 02 — State Management: Reducers, Channels & Merge Strategies

Demonstrates how LangGraph state works:
  1. State is a TypedDict where each key is a 'channel'
  2. Last-Write-Wins (default): plain ``str`` field — latest value wins
  3. Append Reducer: ``Annotated[list, operator.add]`` — values accumulate
  4. Custom Reducer: ``Annotated[int, lambda a, b: a + b]`` — counter

Three analyst nodes (technical, business, ethical) each return a perspective
via the list reducer. Perspectives accumulate automatically — no manual
``.append()`` required.
"""

import operator
from typing import Annotated, TypedDict

from research_langgraph_demo.llm import llm_generate
from langgraph.graph import StateGraph, START, END

# -- State: each key is a 'channel' with its own merge strategy --

class ResearchState(TypedDict):
    topic: str                                          # last-write-wins (default)
    perspectives: Annotated[list, operator.add]         # append reducer
    analysis_count: Annotated[int, lambda a, b: a + b]  # custom reducer (sum)
    status: str                                         # last-write-wins (default)


# -- Nodes: each analyst adds ONE perspective via the list reducer --

def technical_analyst(state: ResearchState) -> dict:
    """Technical perspective — returned list is merged via operator.add."""
    perspective = llm_generate(
        f"In 2-3 sentences, give the technical perspective on {state['topic']} "
        f"focusing on implementation challenges and algorithms."
    )
    return {
        "perspectives": [f"[Technical] {perspective}"],
        "analysis_count": 1,
    }

def business_analyst(state: ResearchState) -> dict:
    """Business perspective — returned list is merged via operator.add."""
    perspective = llm_generate(
        f"In 2-3 sentences, give the business perspective on {state['topic']} "
        f"focusing on market impact and opportunities."
    )
    return {
        "perspectives": [f"[Business] {perspective}"],
        "analysis_count": 1,
    }


def ethical_analyst(state: ResearchState) -> dict:
    """Ethical perspective — returned list is merged via operator.add."""
    perspective = llm_generate(
        f"In 2-3 sentences, give the ethical perspective on {state['topic']} "
        f"focusing on fairness and societal impact."
    )
    return {
        "perspectives": [f"[Ethical] {perspective}"],
        "analysis_count": 1,
    }

def synthesize(state: ResearchState) -> dict:
    """Combine all perspectives into a final summary."""
    combined = "\n".join(state["perspectives"])
    summary = llm_generate(
        f"Synthesize these research perspectives into a brief summary:\n\n{combined}"
    )
    return {
        "perspectives": [f"[Synthesis] {summary}"],
        "analysis_count": 1,
        "status": "synthesis complete",
    }

# -- Build and compile the graph --

def build_graph() -> "CompiledGraph":
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("technical_analyst", technical_analyst)
    graph.add_node("business_analyst", business_analyst)
    graph.add_node("ethical_analyst", ethical_analyst)
    graph.add_node("synthesize", synthesize)

    # Edges: START fans out to all three analysts
    graph.add_edge(START, "technical_analyst")
    graph.add_edge(START, "business_analyst")
    graph.add_edge(START, "ethical_analyst")

    # All analysts converge into synthesize
    graph.add_edge("technical_analyst", "synthesize")
    graph.add_edge("business_analyst", "synthesize")
    graph.add_edge("ethical_analyst", "synthesize")

    # Synthesize -> END
    graph.add_edge("synthesize", END)

    return graph.compile()

def run_demo(topic: str = "AI regulation") -> None:

    app = build_graph()

    initial_state = {
        "topic": topic,
        "perspectives": [],
        "analysis_count": 0,
        "status": "starting",
    }

    print(f"=== State Management Demo: '{topic}' ===\n")
    print("State channels:")
    print("  topic:          str              (last-write-wins)")
    print("  perspectives:   Annotated[list, operator.add]  (append reducer)")
    print("  analysis_count: Annotated[int, lambda a,b: a+b] (custom reducer)")
    print("  status:         str              (last-write-wins)\n")

    # Stream to observe each step
    for step in app.stream(initial_state):
        for node_name, output in step.items():
            print(f"[{node_name}]")
            if "perspectives" in output:
                for p in output["perspectives"]:
                    print(f"  perspective: {p[:100]}...")
            if "analysis_count" in output:
                print(f"  analysis_count delta: +{output['analysis_count']}")
            if "status" in output:
                print(f"  status: {output['status']}")
            print()

    # Final state shows reducer effects
    final = app.invoke(initial_state)
    print("=" * 60)
    print(f"Final status (last-write-wins): {final['status']}")
    print(f"Total analysis_count (custom sum reducer): {final['analysis_count']}")
    print(f"Total perspectives (append reducer): {len(final['perspectives'])} items")
    print()
    for i, p in enumerate(final["perspectives"], 1):
        print(f"  {i}. {p[:120]}...")


if __name__ == "__main__":
    run_demo()
