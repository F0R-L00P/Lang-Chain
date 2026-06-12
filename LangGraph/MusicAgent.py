# A music recommendation agent built on a LangGraph StateGraph + the public
# Deezer API (https://api.deezer.com). Deezer's read endpoints need no API key
# or OAuth, so every tool below is a plain anonymous GET.
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from typing import Annotated, TypedDict


from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()  # expects OPENAI_API_KEY in .env

DEEZER_BASE = "https://api.deezer.com"


def _deezer_get(path, **params):
    """GET {DEEZER_BASE}{path} and return parsed JSON.

    Deezer returns HTTP 200 even for failures, signalling errors with an
    "error" key in the body, so we surface that as an exception. It also rate
    limits to ~50 requests / 5s; on a Quota error we back off once and retry.
    """
    url = f"{DEEZER_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    for attempt in range(3):
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)

        err = data.get("error") if isinstance(data, dict) else None
        if not err:
            return data

        # Quota errors are transient; retry once after a short pause.
        if err.get("code") == 4 and attempt == 0:
            time.sleep(1)
            continue
        raise RuntimeError(f"Deezer API error: {err.get('message', err)}")

    return data


def _resolve_artist(name):
    """Return the best-matching Deezer artist object for a name, or None."""
    hits = _deezer_get("/search/artist", q=name, limit=1).get("data", [])
    return hits[0] if hits else None


# --- Tools the LLM can call -------------------------------------------------


@tool
def search_tracks(query: str, limit: int = 5) -> str:
    """Search Deezer for tracks by free text.

    Supports Deezer's advanced filters inside the query, e.g.
    'artist:"daft punk" track:"around the world"' or 'dur_min:240 rock'.
    Returns a list of "Title - Artist (Album)" AND a 20s preview URL.
    """
    tracks = _deezer_get("/search", q=query, limit=limit).get("data", [])
    if not tracks:
        return f"No tracks found for '{query}'."
    lines = [
        f"- {t['title']} - {t['artist']['name']} "
        f"(album: {t['album']['title']}) | preview: {t.get('preview') or 'n/a'}"
        for t in tracks
    ]
    return "\n".join(lines)


@tool
def find_similar_artists(artist: str, limit: int = 5) -> str:
    """Find artists similar to the given artist name (Deezer /artist/{id}/related)."""
    seed = _resolve_artist(artist)
    if not seed:
        return f"Could not find an artist named '{artist}'."
    related = _deezer_get(f"/artist/{seed['id']}/related", limit=limit).get("data", [])
    if not related:
        return f"No related artists found for {seed['name']}."
    names = ", ".join(a["name"] for a in related)
    return f"Artists similar to {seed['name']}: {names}"


@tool
def get_artist_top_tracks(artist: str, limit: int = 5) -> str:
    """Get an artist's most popular tracks (Deezer /artist/{id}/top)."""
    seed = _resolve_artist(artist)
    if not seed:
        return f"Could not find an artist named '{artist}'."
    tracks = _deezer_get(f"/artist/{seed['id']}/top", limit=limit).get("data", [])
    if not tracks:
        return f"No top tracks found for {seed['name']}."
    lines = [
        f"- {t['title']} (album: {t['album']['title']}) | preview: "
        f"{t.get('preview') or 'n/a'}"
        for t in tracks
    ]
    return f"Top tracks for {seed['name']}:\n" + "\n".join(lines)


@tool
def list_genres() -> str:
    """List Deezer's music genres and their ids (use an id with get_genre_top_artists)."""
    genres = _deezer_get("/genre").get("data", [])
    return ", ".join(f"{g['name']} (id={g['id']})" for g in genres)


@tool
def get_genre_top_artists(genre_id: int, limit: int = 10) -> str:
    """List top artists for a Deezer genre id (Deezer /genre/{id}/artists)."""
    artists = _deezer_get(f"/genre/{genre_id}/artists", limit=limit).get("data", [])
    if not artists:
        return f"No artists found for genre id {genre_id}."
    return ", ".join(a["name"] for a in artists)


# --- Graph ------------------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = (
    "You are a music recommendation assistant powered by the Deezer catalog. "
    "Given an artist, song, genre, or mood, use the tools to discover "
    "similar artists and their popular tracks, then recommend a short, varied "
    "playlist. Prefer find_similar_artists + get_artist_top_tracks for "
    "artist/song seeds, and list_genres + get_genre_top_artists for genre/mood "
    "seeds. Always end with a clean numbered playlist of 'Title - Artist'. "
    "Each recommendation must be followed by a sample preview, if available."
)

tools = [
    search_tracks,
    find_similar_artists,
    get_artist_top_tracks,
    list_genres,
    get_genre_top_artists,
]
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(
    tools
)  # , checkpointer=InMemorySaver()


def agent_node(state: State) -> State:
    return {"messages": [model.invoke(state["messages"])]}


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")  # loop back so the LLM can use the tool result
app = graph.compile()


def visualize_graph(name="music_agent_graph"):
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

    user_input = input("What music are you in the mood for? ")

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]
    for event in app.stream({"messages": messages}, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n=== node: {node_name} ===")
            for msg in node_output["messages"]:
                msg.pretty_print()
