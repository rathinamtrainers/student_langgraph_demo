"""Module 03 — Control Flow: Send, Command & Recursion Limits

Demonstrates three control-flow mechanisms:
  1. Send API  — fan-out: dispatch multiple tasks to the same node with
     different inputs (``[Send("research", {...}) for s in sections]``)
  2. Annotated reducers — ``operator.add`` collects parallel results
  3. Recursion Limit — ``{"recursion_limit": N}`` caps total graph steps;
     exceeding it raises ``GraphRecursionError``

Domain: a ``plan`` node creates research sections, then a conditional edge
uses ``Send`` to fan-out each section to a ``research`` node in parallel.
Results accumulate via ``Annotated[list, operator.add]``.

START => plan =(c)=> research => Aggregate => END

"""

import operator
from typing import Annotated, TypedDict

from research_langgraph_demo.llm import llm_generate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# -- State --

class ResearchState(TypedDict):
    topic: str
    section: str                                    # used by Send per-section
    sections: list                                  # planned section titles
    results: Annotated[list, operator.add]          # collected research results

# -- Nodes --

def plan(state: ResearchState) -> dict:
    """Plan research sections."""
    topic = state["topic"]
    sections = [
        f"Introduction to {topic}",
        f"Technical details of {topic}",
        f"Applications of {topic}",
        f"Future directions for {topic}",
    ]
    return {"sections": sections}

def research(state: ResearchState) -> dict:
    """Research a single section. Results merge via the list reducer."""
    section = state.get("section", "unknown section")
    content = llm_generate(
        f"In 2-3 sentences, summarize key research findings for the "
        f"section: {section}"
    )
    return {
        "results": [{"section": section, "content": content}],
    }


def aggregate(state: ResearchState) -> dict:
    """Combine all section results into a final report."""
    parts = []
    for r in state["results"]:
        parts.append(f"## {r['section']}\n{r['content']}")
    combined = "\n\n".join(parts)

    summary = llm_generate(
        f"Write a one-paragraph executive summary for a research report "
        f"containing these sections:\n\n{combined}"
    )
    return {
        "results": [{"section": "Executive Summary", "content": summary}],
    }


# -- Conditional edge that returns Send objects for fan-out --

def dispatch_sections(state: ResearchState) -> list:
    """Fan-out: return a Send for each planned section."""
    return [
        Send("research", {**state, "section": s})
        for s in state["sections"]
    ]


# -- Build graph --

def build_graph() -> "CompiledGraph":
    graph = StateGraph(ResearchState)

    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("aggregate", aggregate)

    # START -> plan
    graph.add_edge(START, "plan")

    # plan -> conditional edge that dispatches Send objects
    graph.add_conditional_edges("plan", dispatch_sections, ["research"])

    # Each research invocation leads to aggregate
    graph.add_edge("research", "aggregate")

    # Aggregate -> END
    graph.add_edge("aggregate", END)

    return graph.compile()

def run_demo(topic: str = "quantum computing") -> None:

    app = build_graph()

    initial_state = {
        "topic": topic,
        "section": "",
        "sections": [],
        "results": [],
    }

    print(f"=== Control Flow Demo: '{topic}' ===\n")
    print("Key APIs demonstrated:")
    print("  Send        — fan-out: plan dispatches to research nodes in parallel")
    print("  operator.add — list reducer collects parallel results")
    print("  recursion_limit — caps total graph steps\n")

    # Use recursion_limit in config
    config = {"recursion_limit": 25}
    print(f"Config: recursion_limit={config['recursion_limit']}\n")

    # Stream to see each step
    for step in app.stream(initial_state, config=config):
        for node_name, output in step.items():
            if node_name == "plan":
                print(f"[plan] Planned sections:")
                for s in output.get("sections", []):
                    print(f"  - {s}")
                print()
            elif node_name == "research":
                for r in output.get("results", []):
                    print(f"[research] Completed: {r['section']}")
                    print(f"  '{r['content'][:80]}...'\n")
            elif node_name == "aggregate":
                print(f"[aggregate] Combined all results")
                for r in output.get("results", []):
                    print(f"  '{r['content'][:80]}...'\n")

    # Final state
    final = app.invoke(initial_state, config=config)
    print("=" * 60)
    print(f"Total results collected (via reducer): {len(final['results'])}")
    for r in final["results"]:
        print(f"  - {r['section']}")

    print(f"\nRecursion limit was set to {config['recursion_limit']}.")
    print("If the graph exceeded this many steps, LangGraph would raise")
    print("GraphRecursionError to prevent infinite loops.")


if __name__ == "__main__":
    run_demo()


