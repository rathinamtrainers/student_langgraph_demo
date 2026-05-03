"""Module 06 — Multi-Agent: Supervisor Pattern, Command Routing & Subgraphs

Demonstrates multi-agent orchestration in LangGraph:
  1. Supervisor pattern — a central node decides which specialist agent runs next
  2. Command routing  — ``Command(goto='agent', update={...})`` for explicit handoffs
  3. Researcher / Writer / Reviewer agents — each backed by a real LLM call
  4. Subgraph as a node — composing a child graph inside a parent graph
  5. Loop back to supervisor until the review passes or max iterations hit

Graph topology:
  START -> supervisor --(Command)--> researcher -> supervisor
                      --(Command)--> writer     -> supervisor
                      --(Command)--> reviewer   -> supervisor
                      --(Command)--> END
"""
from typing import TypedDict, List

from research_langgraph_demo.llm import llm_generate

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# -- State schema --

class AgentState(TypedDict):
    topic: str
    task: str
    research: str
    draft: str
    review: str
    approved: bool
    iterations: int


# -- Supervisor: decides which agent to invoke next --

def supervisor(state: AgentState) -> Command:
    """Route to the next agent based on what work has been completed."""
    iterations = state.get("iterations", 0)

    # Safety valve: approve after max iterations
    if iterations >= 3:
        print("[supervisor] Max iterations reached — finishing.")
        return Command(goto=END, update={"approved": True})

    if not state.get("research"):
        print(f"[supervisor] Iteration {iterations + 1} — dispatching researcher")
        return Command(goto="researcher", update={"task": "research", "iterations": iterations + 1})

    if not state.get("draft"):
        print(f"[supervisor] Dispatching writer")
        return Command(goto="writer", update={"task": "write"})

    if not state.get("review"):
        print(f"[supervisor] Dispatching reviewer")
        return Command(goto="reviewer", update={"task": "review"})

    # Reviewer left feedback — check approval
    if state.get("approved"):
        print("[supervisor] Review approved — finishing.")
        return Command(goto=END)

    # Not approved: clear draft/review and loop for another iteration
    print(f"[supervisor] Review not approved — restarting cycle (iteration {iterations + 1})")
    return Command(
        goto="researcher",
        update={
            "task": "research",
            "research": "",
            "draft": "",
            "review": "",
            "iterations": iterations + 1,
        },
    )


# -- Agent nodes --

def researcher(state: AgentState) -> dict:
    """Gather research material on the topic."""
    topic = state["topic"]
    previous = state.get("research", "")
    if previous:
        prompt = (
            f"You previously researched '{topic}' and found:\n{previous}\n\n"
            f"Expand on this with additional facts and perspectives. "
            f"Provide 3-4 concise bullet points of new findings."
        )
    else:
        prompt = (
            f"Research the topic '{topic}'. Provide 3-4 concise bullet points "
            f"covering key facts, recent developments, and important context."
        )
    research = llm_generate(prompt)
    print(f"[researcher] Gathered research:\n  {research[:120]}...\n")
    return {"research": research}


def writer(state: AgentState) -> dict:
    """Draft a short report based on the research."""
    prompt = (
        f"Using the following research notes, write a concise 2-3 paragraph report "
        f"on '{state['topic']}'.\n\nResearch notes:\n{state['research']}"
    )
    draft = llm_generate(prompt)
    print(f"[writer] Produced draft:\n  {draft[:120]}...\n")
    return {"draft": draft}


def reviewer(state: AgentState) -> dict:
    """Review the draft and decide whether to approve or request revisions."""
    prompt = (
        f"Review this report draft about '{state['topic']}'. "
        f"If the draft is reasonably clear, accurate, and complete, respond with "
        f"exactly 'APPROVED' on the first line. Otherwise, respond with 'REVISE' "
        f"on the first line followed by specific feedback.\n\n"
        f"Draft:\n{state['draft']}"
    )
    review = llm_generate(prompt)
    approved = review.strip().upper().startswith("APPROVED")
    status = "APPROVED" if approved else "REVISE"
    print(f"[reviewer] Verdict: {status}")
    if not approved:
        print(f"  Feedback: {review[:100]}...\n")
    return {"review": review, "approved": approved}


# -- Build the main graph --

def build_graph() -> "CompiledGraph":
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)

    # Edges: every agent reports back to the supervisor
    graph.add_edge(START, "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


# -- Subgraph demo: embed a research sub-pipeline inside a parent graph --

class ParentState(TypedDict):
    topic: str
    summary: str


class ResearchSubState(TypedDict):
    topic: str
    research: str


def sub_research(state: ResearchSubState) -> dict:
    research = llm_generate(
        f"Provide 2-3 key facts about '{state['topic']}' as bullet points."
    )
    return {"research": research}


def sub_summarize(state: ResearchSubState) -> dict:
    summary = llm_generate(
        f"Summarize these research notes in one sentence:\n{state['research']}"
    )
    return {"research": summary}


def build_research_subgraph():
    """Build a small subgraph: research -> summarize."""
    sg = StateGraph(ResearchSubState)
    sg.add_node("sub_research", sub_research)
    sg.add_node("sub_summarize", sub_summarize)
    sg.add_edge(START, "sub_research")
    sg.add_edge("sub_research", "sub_summarize")
    sg.add_edge("sub_summarize", END)
    return sg.compile()


def build_parent_graph() -> "CompiledGraph":
    """Parent graph that uses a compiled subgraph as a node."""
    research_subgraph = build_research_subgraph()

    def format_output(state: ParentState) -> dict:
        return {"summary": f"Report on '{state['topic']}': {state.get('summary', 'N/A')}"}

    parent = StateGraph(ParentState)
    # Add the compiled subgraph directly as a node
    parent.add_node("research", research_subgraph)
    parent.add_node("format", format_output)
    parent.add_edge(START, "research")
    parent.add_edge("research", "format")
    parent.add_edge("format", END)
    return parent.compile()


# -- Entry point --

def run_demo(topic: str = "sustainable energy") -> None:

    # --- Part 1: Supervisor pattern with Command routing ---
    print("=" * 60)
    print("PART 1: Supervisor Pattern with Command Routing")
    print("=" * 60)

    app = build_graph()
    initial_state: AgentState = {
        "topic": topic,
        "task": "",
        "research": "",
        "draft": "",
        "review": "",
        "approved": False,
        "iterations": 0,
    }

    final = app.invoke(initial_state)
    print("\n" + "=" * 60)
    print(f"Final approved: {final['approved']}")
    print(f"Iterations:     {final['iterations']}")
    print(f"Draft preview:  {final.get('draft', '')[:200]}")

    # --- Part 2: Subgraph as a node ---
    print("\n" + "=" * 60)
    print("PART 2: Subgraph as a Node")
    print("=" * 60)

    parent_app = build_parent_graph()
    result = parent_app.invoke({"topic": topic, "summary": ""})
    print(f"Parent graph result: {result.get('summary', '')[:200]}")


if __name__ == "__main__":
    run_demo()