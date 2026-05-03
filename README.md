# LangChain Search Engine

A conversational search assistant built with LangChain, LangGraph, Groq, and Streamlit. The agent can search the web, Wikipedia, and Arxiv research papers to answer your questions, and tells you which source was used for each answer.

---

## Features

- Conversational chat interface built with Streamlit
- Searches three sources: DuckDuckGo (web), Wikipedia, and Arxiv
- Shows which source was used for each answer
- Uses Groq's `openai/gpt-oss-120b` reasoning model
- Persistent chat history within the session

---

## Project Structure

```
.
├── app.py          # Main Streamlit application
├── .env            # API keys (not committed to version control)
└── README.md       # This file
```

---

## Requirements

- Python 3.9 or higher
- A Groq API key — get one free at https://console.groq.com

---

## Installation

1. Clone or download the project folder.

2. Create and activate a virtual environment:

```bash
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On Mac/Linux
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install streamlit langchain langchain-groq langchain-community langgraph python-dotenv
```

---

## Configuration

Create a `.env` file in the root of the project folder and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key from https://console.groq.com/keys.

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How It Works

The app uses a LangGraph ReAct agent that decides which tool to use based on your question.

- For research paper queries, it uses the **Arxiv** tool
- For general knowledge queries, it uses the **Wikipedia** tool
- For current events and web searches, it uses the **DuckDuckGo Search** tool
- If the model can answer from its own knowledge, no tool is called

After each response, the app displays a `Source:` label showing which tool or tools were used.

---

## Model

The app uses `openai/gpt-oss-120b` hosted on Groq. This is a reasoning model so streaming is disabled — a spinner is shown while the agent is working instead.

If you want to switch models, replace the `model_name` in `app.py`:

```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",  # change this
    streaming=False
)
```

Available Groq production models as of May 2026:

| Model ID | Speed | Notes |
|---|---|---|
| `openai/gpt-oss-120b` | 500 t/s | Highest quality, reasoning model |
| `openai/gpt-oss-20b` | 1000 t/s | Faster, lower cost |
| `llama-3.3-70b-versatile` | 280 t/s | Good tool calling |
| `llama-3.1-8b-instant` | 560 t/s | Fastest, lightweight |

Note: If you switch to a non-reasoning model you can set `streaming=True`.

---

## Known Issues

- `openai/gpt-oss-120b` and other reasoning models must use `streaming=False`. Setting `streaming=True` causes a `RecursionError` in the Streamlit callback handler due to internal thought tokens emitted by reasoning models.
- DuckDuckGo search may occasionally be rate limited. If you see DuckDuckGo errors, wait a few seconds and try again.
- Groq has a daily rate limit on the free developer plan. If you hit the limit, wait for it to reset or switch to a lower-cost model.

---

## License

MIT
