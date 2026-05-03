# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A teaching-oriented collection of LangGraph demos. Each module under `src/research_langgraph_demo/demos/` illustrates one LangGraph concept (state, control flow, persistence, multi-agent, etc.) with a self-contained graph and a `run_demo()` entrypoint.

## Environment & Setup

- Python project using `src/` layout — anything that runs the package must put `src` on `PYTHONPATH`.
- The `.venv/` in the repo root is the project virtualenv. Activate with `source .venv/Scripts/activate` (Git Bash) or `.venv\Scripts\Activate.ps1` (PowerShell).
- Install deps: `pip install -r requirements.txt`.
- LLM credentials come from `.env` at repo root. `GOOGLE_API_KEY` is loaded via `python-dotenv` inside `llm.py` — no shell export needed.

## Common Commands

Run a single demo (preferred — they are independent):

```bash
PYTHONPATH=src python -m research_langgraph_demo.demos.core_concepts
PYTHONPATH=src python -m research_langgraph_demo.demos.persistence
# etc. — one of the demo module names listed below
```

The VS Code launch config in `.vscode/launch.json` runs `research_langgraph_demo.demos.streaming` with `PYTHONPATH=src` — duplicate that block to debug other demos.

There is no test suite, no linter config, and no build step.

## Architecture

### LLM provider layer (`src/research_langgraph_demo/llm.py`)

Single chokepoint for LLM access. Exports `get_llm()`, `llm_generate()`, `llm_agenerate()`, and the `HAS_GEMINI` flag.

- Primary path: `ChatGoogleGenerativeAI` (Gemini, model `gemini-2.0-flash`) when `langchain-google-genai` is importable AND `GOOGLE_API_KEY` is set.
- Fallback path: an in-file `MockLLM` class used when the Gemini import fails. **Note:** the fallback only triggers on `ImportError`, not on a missing API key — `get_llm()` raises `RuntimeError` if Gemini is installed but the key is absent. Some demos (e.g. `core_concepts.run_demo`) additionally guard on `HAS_GEMINI` and refuse to run without a real LLM.

When adding new code that needs the LLM, import from this module rather than instantiating providers directly. This keeps the Gemini-vs-mock decision in one place.

### Demo modules (`src/research_langgraph_demo/demos/`)

Each demo is a standalone file with the same shape:

1. A `TypedDict` state schema at the top of the file.
2. Plain-function nodes that take state and return a dict patch.
3. A `build_graph()` that wires nodes/edges and returns the compiled graph.
4. A `run_demo(...)` that constructs initial state and streams/invokes the graph.
5. An `if __name__ == "__main__": run_demo()` block.

Stick to this shape when adding a new demo — `demos/__init__.py` and the IDE configs assume it.

**Important caveat about `demos/__init__.py`:** it imports a long list of demo modules (`prebuilt_components`, `long_term_memory`, `time_travel`, `functional_api`, `async_execution`, `map_reduce`, `dynamic_breakpoints`, `platform_concepts`, `subgraph_communication`, `custom_reducers`, `fault_tolerance`, `langsmith_tracing`, `tool_calling_patterns`, `graph_introspection`) that **do not yet exist on disk**. As a result, `import research_langgraph_demo.demos` and `demos.run_all()` currently raise `ImportError`. Run individual demo modules directly (as shown above) instead of going through the package. When adding one of the missing modules, follow the standard demo shape and the package import will start working for that name.

### Cross-demo conventions worth knowing

- State is always a `TypedDict`. Reducers are attached via `Annotated[T, reducer]` (`operator.add` for list-append, custom lambdas for counters). See `state_management.py` for the canonical example.
- Multi-agent demos use `langgraph.types.Command(goto=..., update=...)` for explicit handoffs from a supervisor node, instead of conditional edges. See `multi_agent.py`.
- Persistence demos use `MemorySaver` from `langgraph.checkpoint.memory` and pass `config={"configurable": {"thread_id": ...}}` to `invoke`/`stream`. See `persistence.py`.
- Demos that show streaming iterate `app.stream(initial_state)` and inspect the per-node dict yielded each step.

## Things to avoid

- Do not commit `.env` — it contains a live `GOOGLE_API_KEY`.
- Do not import Vertex AI; the project deliberately uses `langchain-google-genai` (Gemini) only, per project preference baked into `llm.py`'s docstring.
- Do not "fix" the broken imports in `demos/__init__.py` by deleting the missing module names — the list is the planned curriculum. Add the missing module instead, or run demos directly.
