"""Demo modules package

Expose run_all() which executes each demo's run_demo() in order.
"""
from . import (
    core_concepts,
    state_management,
    control_flow,
    persistence,
    human_in_the_loop,
    multi_agent,
    streaming,
    prebuilt_components,
    long_term_memory,
    time_travel,
    functional_api,
    async_execution,
    map_reduce,
    dynamic_breakpoints,
    platform_concepts,
    subgraph_communication,
    custom_reducers,
    fault_tolerance,
    langsmith_tracing,
    tool_calling_patterns,
    graph_introspection,
)

modules = [
    core_concepts,
    state_management,
    control_flow,
    persistence,
    human_in_the_loop,
    multi_agent,
    streaming,
    prebuilt_components,
    long_term_memory,
    time_travel,
    functional_api,
    async_execution,
    map_reduce,
    dynamic_breakpoints,
    platform_concepts,
    subgraph_communication,
    custom_reducers,
    fault_tolerance,
    langsmith_tracing,
    tool_calling_patterns,
    graph_introspection,
]


def run_all():
    for m in modules:
        print(f"\n--- Running {m.__name__.split('.')[-1]} ---")
        try:
            m.run_demo()
        except Exception as e:
            print(f"Demo {m.__name__} failed: {e}")

__all__ = ["run_all"]
