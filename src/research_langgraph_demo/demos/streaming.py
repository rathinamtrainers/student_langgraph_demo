"""Module 07 — Streaming: Values, Updates & Debug Modes

Demonstrates LangGraph's built-in streaming capabilities:
  1. ``stream_mode="values"``  — full state snapshot emitted after each node
  2. ``stream_mode="updates"`` — only the delta (changed keys) per node step
  3. ``stream_mode="debug"``   — verbose events with timing and metadata
  4. Multiple modes simultaneously via list of mode names

Graph topology (simple three-step pipeline):
  START -> research -> draft -> review -> END
"""
from typing import TypedDict

from research_langgraph_demo.llm import llm_generate

from langgraph.graph import StateGraph, START, END


# -- State schema --

class ReportState(TypedDict):
    topic: str
    research: str
    draft: str
    review: str
    step_count: int


# -- Nodes --

def research(state: ReportState) -> dict:
    """Gather research facts on the topic."""
    content = llm_generate(
        f"Provide 3 concise bullet points of key facts about '{state['topic']}'."
    )
    return {"research": content, "step_count": state.get("step_count", 0) + 1}


def draft(state: ReportState) -> dict:
    """Write a short report draft from the research."""
    content = llm_generate(
        f"Using these research notes, write a concise one-paragraph report "
        f"about '{state['topic']}':\n\n{state['research']}"
    )
    return {"draft": content, "step_count": state.get("step_count", 0) + 1}


def review(state: ReportState) -> dict:
    """Provide a brief quality review of the draft."""
    content = llm_generate(
        f"In 1-2 sentences, assess the quality of this draft report:\n\n{state['draft']}"
    )
    return {"review": content, "step_count": state.get("step_count", 0) + 1}


# -- Build the graph --

def build_graph() -> "CompiledGraph":
    graph = StateGraph(ReportState)

    graph.add_node("research", research)
    graph.add_node("draft", draft)
    graph.add_node("review", review)

    graph.add_edge(START, "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", END)

    return graph.compile()


# -- Entry point --

def run_demo(topic: str = "renewable energy trends") -> None:

    app = build_graph()
    initial_state: ReportState = {
        "topic": topic,
        "research": "",
        "draft": "",
        "review": "",
        "step_count": 0,
    }

    # ---------------------------------------------------------
    # 1. stream_mode="values" — full state snapshot each step
    # ---------------------------------------------------------
    print("=" * 60)
    print('STREAM MODE: "values" — full state snapshot after each node')
    print("=" * 60)

    for state_snapshot in app.stream(initial_state, stream_mode="values"):
        # Each emission is the complete state dict
        step = state_snapshot.get("step_count", 0)
        print(f"\n[values] step_count={step}")
        for key, value in state_snapshot.items():
            if key == "topic":
                continue
            preview = str(value)[:100] if isinstance(value, str) and len(str(value)) > 100 else value
            print(f"  {key}: {preview}")

    # ---------------------------------------------------------
    # 2. stream_mode="updates" — only the delta per node
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print('STREAM MODE: "updates" — only changed keys per node step')
    print("=" * 60)

    for update in app.stream(initial_state, stream_mode="updates"):
        # Each emission is {node_name: {changed_keys}}
        for node_name, changes in update.items():
            print(f"\n[updates] Node: {node_name}")
            for key, value in changes.items():
                preview = str(value)[:100] if isinstance(value, str) and len(str(value)) > 100 else value
                print(f"  {key}: {preview}")

    # ---------------------------------------------------------
    # 3. stream_mode="debug" — verbose events with metadata
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print('STREAM MODE: "debug" — verbose events with metadata')
    print("=" * 60)

    for event in app.stream(initial_state, stream_mode="debug"):
        event_type = event.get("type", "unknown")
        step = event.get("step", "?")
        # Show a compact summary of each debug event
        if event_type == "task":
            node = event.get("payload", {}).get("name", "?")
            print(f"\n[debug] step={step} type={event_type} node={node}")
        elif event_type == "task_result":
            node = event.get("payload", {}).get("name", "?")
            print(f"[debug] step={step} type={event_type} node={node}")
        else:
            print(f"[debug] step={step} type={event_type}")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print('  "values"  — full state after each node (good for UI state display)')
    print('  "updates" — delta only (efficient for incremental UI updates)')
    print('  "debug"   — verbose internals (good for development/debugging)')
    print('  "messages"— token-by-token LLM output (requires ChatModel integration)')


if __name__ == "__main__":
    run_demo()