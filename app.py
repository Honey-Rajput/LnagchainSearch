import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------
# Tool Setup
# -------------------------------------------------------

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = DuckDuckGoSearchRun(name="Search")

tools = [arxiv_tool, wiki_tool, search_tool]

# System message injected into every invoke call.
# Works across all LangGraph versions without
# using state_modifier or prompt arguments.
system_message = SystemMessage(content="""You are a helpful search assistant.
You have access to exactly three tools:
1. arxiv - for searching research papers
2. wikipedia - for searching Wikipedia articles
3. Search - for general web searches using DuckDuckGo

Do not call any other tools. Never call brave_search or any tool not listed above.
Only call one tool at a time. Keep your tool inputs simple and concise.
""")

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(page_title="LangChain Search Agent", layout="centered")
st.title("LangChain Search Engine")
st.caption("Ask me anything. I can search the web, Wikipedia, and research papers.")

# -------------------------------------------------------
# Session State
# -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hello! I am a search assistant. I can look things up on the web, Wikipedia, and Arxiv. What would you like to know?"
        }
    ]

# Display all previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------------------------------
# Chat Input
# -------------------------------------------------------

prompt = st.chat_input(placeholder="Ask something, e.g. What is machine learning?")

if prompt:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Validate API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
        st.stop()

    # -------------------------------------------------------
    # LLM Setup
    # streaming=False is required for openai/gpt-oss-120b
    # because it is a reasoning model. Streaming causes
    # RecursionError in StreamlitCallbackHandler with
    # reasoning models that emit internal thought tokens.
    # -------------------------------------------------------

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-120b",
        streaming=False
    )

    search_agent = create_react_agent(llm, tools)

    with st.chat_message("assistant"):
        # Show a spinner while the agent is working
        # since we are not streaming token by token
        with st.spinner("Agent is thinking..."):
            try:
                response = search_agent.invoke(
                    {"messages": [system_message, {"role": "user", "content": prompt}]}
                )

                # -------------------------------------------------------
                # Detect which tools were used during the agent run
                # -------------------------------------------------------

                tools_used = []
                for msg in response["messages"]:
                    if hasattr(msg, "name") and msg.name:
                        tools_used.append(msg.name)

                unique_tools = list(dict.fromkeys(tools_used))

                if unique_tools:
                    source_text = ", ".join(unique_tools)
                    st.caption(f"Source: {source_text}")
                else:
                    st.caption("Source: LLM (no tools used, answered from model knowledge)")

                st.caption("Model: openai/gpt-oss-120b")

                # Get the final answer from the last message
                final_answer = response["messages"][-1].content

                # Save to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

                # Display the final answer
                st.write(final_answer)

            except Exception as e:
                error_message = f"Something went wrong: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )