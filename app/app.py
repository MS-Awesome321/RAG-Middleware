import sys
import os
sys.path.append(os.path.join(os.pardir, 'rag'))
from rag import RAG
from agent import response_generator
import streamlit as st

st.title("RAG with ChatGPT, Pinecone DB, and Reranker")

rag = RAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": rag.augment_prompt(prompt)})

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response})