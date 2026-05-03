"""Module 05 — Human-in-the-Loop: interrupt() and Command(resume=...)

Demonstrates the LangGraph interrupt/resume pattern:
  1. interrupt()          — pauses execution, serializes state to checkpoint
  2. Command(resume=val)  — resumes the graph with the human's decision
  3. Requires a checkpointer (MemorySaver) so state survives the pause

The graph: START -> draft -> review -> finalize -> END

The review node calls interrupt() to pause for human approval. The caller
resumes with Command(resume="Looks good!") to continue to finalize.

For testability the resume value is read from os.getenv("HUMAN_RESPONSE").
"""
import os
from typing import TypedDict

from research_langgraph_demo.llm import llm_generate


from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command


# -- State --

class ReviewState(TypedDict):
    topic: str
    draft: str
    human_feedback: str
    final_report: str


# -- Nodes --

def draft_node(state: ReviewState) -> dict:
    """Generate an initial draft via LLM."""
    content = llm_generate(
        f"Write a short draft report (2-3 sentences) about {state['topic']}."
    )
    return {"draft": content}


def review_node(state: ReviewState) -> dict:
    """Pause for human review using interrupt().

    interrupt() serializes the current state to the checkpoint and halts
    execution. The caller resumes later with Command(resume=<value>).
    """
    human_input = interrupt({
        "draft": state["draft"],
        "prompt": "Please review this draft. Approve or provide feedback.",
    })
    return {"human_feedback": human_input}


def finalize_node(state: ReviewState) -> dict:
    """Produce the final report incorporating human feedback."""
    final = llm_generate(
        f"Finalize this report about {state['topic']}. "
        f"Original draft: {state['draft']}. "
        f"Human feedback: {state['human_feedback']}."
    )
    return {"final_report": final}


# -- Build graph --

def build_graph():
    """Build and compile the review pipeline with MemorySaver."""
    workflow = StateGraph(ReviewState)

    workflow.add_node("draft", draft_node)
    workflow.add_node("review", review_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "review")
    workflow.add_edge("review", "finalize")
    workflow.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def run_demo() -> None:

    app = build_graph()
    config = {"configurable": {"thread_id": "review-session-1"}}

    initial = {"topic": "AI safety", "draft": "", "human_feedback": "", "final_report": ""}

    # --- 1. First invoke: runs until interrupt() in review node ---
    print("=== Human-in-the-Loop Demo ===\n")
    print("--- Invoke 1: runs until interrupt ---")
    result = app.invoke(initial, config)
    print(f"  Draft generated: {result.get('draft', '')[:120]}...")

    # Check that the graph is paused at review
    state_snapshot = app.get_state(config)
    print(f"  Graph paused at: {state_snapshot.next}")
    print(f"  (interrupt payload available for human review)\n")

    # --- 2. Resume with human feedback via Command(resume=...) ---
    human_response = os.getenv("HUMAN_RESPONSE", "approve")
    resume_value = "Looks good! Please finalize." if human_response == "approve" else human_response
    print(f"--- Resume with Command(resume='{resume_value}') ---")
    result = app.invoke(Command(resume=resume_value), config)
    print(f"  Human feedback: {result.get('human_feedback', '')}")
    print(f"  Final report:   {result.get('final_report', '')[:120]}...\n")

    # --- 3. Verify final state ---
    final_snapshot = app.get_state(config)
    print("--- Final state ---")
    print(f"  Next: {final_snapshot.next} (empty = graph completed)")
    print(f"  Keys: {list(final_snapshot.values.keys())}")


if __name__ == "__main__":
    run_demo()

