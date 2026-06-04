# A simple weather agent built explicitly from a LangGraph StateGraph + nodes.
import json
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

load_dotenv()  # expects OPENAI_API_KEY in .env


def _get(base, **params):
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Input is a plain city name, e.g. "Tokyo"."""
    geo = _get("https://geocoding-api.open-meteo.com/v1/search", name=city, count=1)
    if not geo.get("results"):
        return f"Could not find a location named '{city}'."

    place = geo["results"][0]
    w = _get(
        "https://api.open-meteo.com/v1/forecast",
        latitude=place["latitude"],
        longitude=place["longitude"],
        current_weather="true",
    )["current_weather"]
    return (
        f"Weather in {place['name']}, {place.get('country', '')}: "
        f"{w['temperature']}°C, wind {w['windspeed']} km/h."
    )


# Graph state: a running list of messages that nodes append to.
class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [get_weather]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# The LLM decides whether to answer or call a tool.
def agent_node(state: State) -> State:
    return {"messages": [model.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
# tools_condition routes to "tools" if the last message asked for one, else END.
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")  # loop back so the LLM can use the tool result
app = graph.compile()


def visualize_graph(name="agent_graph"):
    """Save the compiled graph as Mermaid text (offline) and, if reachable, a PNG."""
    base = Path(__file__).parent / name
    g = app.get_graph()
    base.with_suffix(".mmd").write_text(g.draw_mermaid())
    print(f"Saved Mermaid diagram to {base}.mmd")
    try:  # PNG needs the mermaid.ink web API; skip if it isn't reachable.
        base.with_suffix(".png").write_bytes(g.draw_mermaid_png())
        print(f"Saved PNG diagram to {base}.png")
    except Exception as e:
        print(f"Skipped PNG render ({type(e).__name__}); .mmd is available.")


if __name__ == "__main__":
    visualize_graph()

    user_input = input("target weather input: ")

    # stream_mode="updates" yields one event per node as it runs, so we can
    # watch the agent decide, the tool execute, and the agent answer in order.
    for event in app.stream({"messages": [HumanMessage(content=user_input)]}, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n=== node: {node_name} ===")
            for msg in node_output["messages"]:
                msg.pretty_print()