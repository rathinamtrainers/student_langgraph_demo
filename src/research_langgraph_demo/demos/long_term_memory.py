"""Module 09 — Long-Term Memory: InMemoryStore & Cross-Thread Recall

Demonstrates the distinction between checkpointers and stores:
  - Checkpointer (MemorySaver) = within-thread state persistence
  - Store (InMemoryStore)      = cross-thread persistent memory

Key APIs:
  store.put(namespace_tuple, key, value_dict)  — save a fact
  store.search(namespace_tuple)                — retrieve memories
  Store is injected into nodes via the `store` parameter in the signature.

Real LangGraph pattern:
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver

    store = InMemoryStore()
    saver = MemorySaver()
    graph = workflow.compile(checkpointer=saver, store=store)
"""
from typing import Annotated, TypedDict

from research_langgraph_demo.llm import llm_generate

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


# -- State --

class MemoryState(TypedDict):
    user_id: str
    query: str
    response: str


# -- Nodes --

def save_preferences(state: MemoryState, *, store) -> dict:
    """Save user preferences to the cross-thread store."""
    user_id = state["user_id"]
    query = state["query"]

    # Derive a preference from the query
    preference = llm_generate(
        f"Extract the user preference from this message as a short phrase: {query}"
    )

    store.put(("users", user_id, "preferences"), "latest_pref", {"value": preference})
    store.put(("users", user_id, "preferences"), "format_pref", {"value": "concise summaries"})

    return {"response": f"Saved preference for {user_id}: {preference}"}


def recall_and_respond(state: MemoryState, *, store) -> dict:
    """Recall stored memories and generate a personalized response."""
    user_id = state["user_id"]
    query = state["query"]

    # Search for all memories in this user's namespace
    memories = store.search(("users", user_id, "preferences"))

    if memories:
        mem_text = "; ".join(
            f"{item.key}={item.value}" for item in memories
        )
        response = llm_generate(
            f"The user previously expressed these preferences: {mem_text}. "
            f"Now answer their query in a way that respects those preferences: {query}"
        )
    else:
        response = llm_generate(f"Answer this query: {query}")

    return {"response": response}


# -- Build graphs --

def build_save_graph(store, saver):
    """Build a graph that saves user preferences to the store."""
    workflow = StateGraph(MemoryState)
    workflow.add_node("save_preferences", save_preferences)
    workflow.add_edge(START, "save_preferences")
    workflow.add_edge("save_preferences", END)
    return workflow.compile(checkpointer=saver, store=store)


def build_recall_graph(store, saver):
    """Build a graph that recalls memories and responds."""
    workflow = StateGraph(MemoryState)
    workflow.add_node("recall_and_respond", recall_and_respond)
    workflow.add_edge(START, "recall_and_respond")
    workflow.add_edge("recall_and_respond", END)
    return workflow.compile(checkpointer=saver, store=store)


def run_demo() -> None:

    # Shared store (cross-thread) and checkpointer (per-thread)
    store = InMemoryStore()
    saver = MemorySaver()

    save_app = build_save_graph(store, saver)
    recall_app = build_recall_graph(store, saver)

    print("=== Long-Term Memory Demo ===\n")

    # --- 1. Save preferences on thread-1 ---
    print("--- 1. Save user preferences (thread-1) ---")
    config_1 = {"configurable": {"thread_id": "thread-1"}}
    result_1 = save_app.invoke(
        {"user_id": "alice", "query": "I prefer detailed technical explanations", "response": ""},
        config_1,
    )
    print(f"  {result_1['response']}\n")

    # --- 2. Verify store contents directly ---
    print("--- 2. Verify store contents ---")
    items = store.search(("users", "alice", "preferences"))
    for item in items:
        print(f"  key={item.key}, value={item.value}")
    print()

    # --- 3. Recall on a DIFFERENT thread (cross-thread memory) ---
    print("--- 3. Cross-thread recall (thread-2) ---")
    config_2 = {"configurable": {"thread_id": "thread-2"}}
    result_2 = recall_app.invoke(
        {"user_id": "alice", "query": "Explain how neural networks learn", "response": ""},
        config_2,
    )
    print(f"  Response: {result_2['response'][:200]}...\n")

    # --- 4. New user with no stored memories ---
    print("--- 4. New user with no memories (thread-3) ---")
    config_3 = {"configurable": {"thread_id": "thread-3"}}
    result_3 = recall_app.invoke(
        {"user_id": "bob", "query": "Explain how neural networks learn", "response": ""},
        config_3,
    )
    print(f"  Response: {result_3['response'][:200]}...\n")

    # --- 5. Show that store is cross-thread but checkpointer is per-thread ---
    print("--- 5. Checkpointer vs Store ---")
    print(f"  Store memories for alice: {len(store.search(('users', 'alice', 'preferences')))} items")
    print(f"  Store memories for bob:   {len(store.search(('users', 'bob', 'preferences')))} items")
    snap_1 = save_app.get_state(config_1)
    snap_2 = recall_app.get_state(config_2)
    print(f"  thread-1 query: {snap_1.values.get('query', 'N/A')}")
    print(f"  thread-2 query: {snap_2.values.get('query', 'N/A')}")
    print(f"  Checkpointer isolates threads; Store spans all threads.")


if __name__ == "__main__":
    run_demo()