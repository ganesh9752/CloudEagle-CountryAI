from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import intent_node, tool_node, synthesis_node, error_node


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def _route_after_intent(state: AgentState) -> str:
    """Go to tool if countries were found, otherwise go to error."""
    if state.get("error") or not state.get("countries"):
        return "error"
    return "tool"


def _route_after_tool(state: AgentState) -> str:
    """Go to synthesis if we have any data, otherwise go to error."""
    if not state.get("raw_data"):
        return "error"
    return "synthesis"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("error", error_node)

    graph.set_entry_point("intent")

    graph.add_conditional_edges(
        "intent",
        _route_after_intent,
        {"tool": "tool", "error": "error"},
    )

    graph.add_conditional_edges(
        "tool",
        _route_after_tool,
        {"synthesis": "synthesis", "error": "error"},
    )

    graph.add_edge("synthesis", END)
    graph.add_edge("error", END)

    return graph.compile()


# Module-level singleton — compiled once at import time.
agent = build_graph()
