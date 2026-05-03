"""Module 04 — Persistence: Checkpointers, MemorySaver & thread_id

Demonstrates how LangGraph checkpointers save full graph state at every step:
  1. MemorySaver  — in-process dict-based checkpointer (dev/testing)
  2. thread_id    — config key that isolates independent conversations
  3. get_state()  — inspect the current checkpoint for a thread

The graph: START -> research -> draft -> END

Two threads are run to show that thread_id keeps conversations independent,
and a second invocation on the same thread shows state resumption.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from research_langgraph_demo.llm import llm_generate


# -- State --

class ResearchState(TypedDict):
    topic: str
    notes: str
    draft: str


# -- Nodes --

def research(state: ResearchState) -> dict:
    """Gather research notes on the topic."""
    notes = llm_generate(
        f"In 2-3 sentences, note key findings about {state['topic']}."
    )
    return {"notes": notes}


def draft(state: ResearchState) -> dict:
    """Write a short draft report from the research notes."""
    report = llm_generate(
        f"Write a one-paragraph draft report about {state['topic']} "
        f"based on these notes: {state['notes']}"
    )
    return {"draft": report}

# -- Build graph --

def build_graph():
    """Build and compile a research pipeline with MemorySaver checkpointer."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("research", research)
    workflow.add_node("draft", draft)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "draft")
    workflow.add_edge("draft", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_demo() -> None:

    app = build_graph()

    # --- 1. First invocation on thread-1 ---
    config_1 = {"configurable": {"thread_id": "thread-1"}}
    initial = {"topic": "climate change", "notes": "", "draft": ""}

    print("=== Persistence Demo ===\n")
    print("--- Invoke 1: thread-1 (new session) ---")
    result_1 = app.invoke(initial, config_1)
    print(f"  Topic: {result_1['topic']}")
    print(f"  Draft: {result_1['draft'][:120]}...\n")

    # --- 2. Inspect state via get_state() ---
    print("--- get_state(thread-1) ---")
    state_snapshot = app.get_state(config_1)
    print(f"  Values keys: {list(state_snapshot.values.keys())}")
    print(f"  Next:        {state_snapshot.next}")
    print(f"  Thread ID:   {state_snapshot.config['configurable']['thread_id']}\n")

    # --- 3. Second invocation on same thread (state resumes) ---
    print("--- Invoke 2: thread-1 (same thread, state resumes) ---")
    result_2 = app.invoke(initial, config_1)
    print(f"  Draft: {result_2['draft'][:120]}...\n")

    # --- 4. Different thread_id = independent conversation ---
    config_2 = {"configurable": {"thread_id": "thread-2"}}
    initial_2 = {"topic": "quantum computing", "notes": "", "draft": ""}

    print("--- Invoke 3: thread-2 (independent conversation) ---")
    result_3 = app.invoke(initial_2, config_2)
    print(f"  Topic: {result_3['topic']}")
    print(f"  Draft: {result_3['draft'][:120]}...\n")

    # --- 5. Verify threads are independent ---
    print("--- Verify thread isolation ---")
    snap_1 = app.get_state(config_1)
    snap_2 = app.get_state(config_2)
    print(f"  thread-1 topic: {snap_1.values['topic']}")
    print(f"  thread-2 topic: {snap_2.values['topic']}")
    print(f"  Threads are independent: {snap_1.values['topic'] != snap_2.values['topic']}")

    print(f" snap_1 = {snap_1}")
    print(f" snap_2 = {snap_2}")

if __name__ == "__main__":
    run_demo()


