# app.py
# import asyncio
# import sys

# if sys.platform.startswith("linux") and sys.version_info >= (3, 10):
#     try:
#         asyncio.get_event_loop()
#     except RuntimeError:
#         asyncio.set_event_loop(asyncio.new_event_loop())

import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
from chatbot_core import chat_agent

st.set_page_config(page_title="Web Search AI Chatbot", page_icon="🤖", layout="wide")
st.title("🔍 Agentic AI Chatbot")
st.markdown("Ask me anything. I’ll pull answers from the web and cite the sources")

import torch

if not hasattr(torch, "get_default_device"):
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = get_default_device

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

#chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
query = st.chat_input("Type your question here...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Searching the web and thinking..."):
        answer = chat_agent(query, embedder)
    st.session_state.messages.append({"role": "ai", "content": answer})
#display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#chat history
# if "history" not in st.session_state:
#     st.session_state.history = []
# if query:
#     with st.spinner("Searching the web and thinking..."):
#         answer = chat_agent(query, embedder)
#     st.session_state.history.append(("You", query))
#     st.session_state.history.append(("AI", answer))
# for speaker, message in st.session_state.history:
#     if speaker == "You":
#         st.markdown(f"**You:** {message}")
#     else:
#         st.markdown(f"**AI:** {message}")