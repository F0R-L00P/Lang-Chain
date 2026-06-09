# Streamlit chat UI for the LangGraph music agent in MusicAgent.py.
# Run with:  streamlit run LangGraph/MusicAgentUI.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from MusicAgent import SYSTEM_PROMPT, app

st.set_page_config(page_title="Music Agent", page_icon="🎵")
st.title("🎵 Music Recommendation Agent")
st.caption("Powered by LangGraph + the Deezer catalog. Ask for a seed artist, song, genre, or mood.")

# Conversation history lives in session state. We keep the SystemMessage at the
# front so the agent stays in character across turns, but never render it.
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]


def render_message(msg):
    """Draw a single LangChain message in the chat transcript (skip empty/tool noise)."""
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        with st.chat_message("assistant"):
            with st.expander(f"🔧 tool result · {msg.name}", expanded=False):
                st.text(msg.content)


# Replay the transcript so far (everything except the system prompt).
for msg in st.session_state.messages[1:]:
    render_message(msg)

if prompt := st.chat_input("What music are you in the mood for?"):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # Stream the graph and render each new node output as it arrives. We track how
    # many messages we've already appended so we only render the fresh ones.
    with st.spinner("Thinking..."):
        for event in app.stream({"messages": st.session_state.messages}, stream_mode="updates"):
            for node_output in event.values():
                for msg in node_output["messages"]:
                    st.session_state.messages.append(msg)
                    render_message(msg)
