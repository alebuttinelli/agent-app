# Language Simplification Agent

This is a web application, built with [Streamlit](https://streamlit.io/), that acts as an **expert consultant for simplifying administrative and bureaucratic language**.

A user can paste in a text, and the AI agent will analyse it, identifying the main categories of issues (e.g., long sentences, passive voice, jargon) and providing specific advice for improvement.

## Key Features

* **Simple Web Interface:** Built with Streamlit for easy input and output.
* **AI Agent-Based Analysis:** Uses a LangChain ReAct (Reasoning + Acting) agent to analyse the text in a structured way.
* **Custom Knowledge Base (RAG):** The agent doesn't invent rules! It uses a *Retrieval-Augmented Generation* (RAG) tool to search for simplification rules from a knowledge file (`regole.txt`).
* **Powerful Models:** Leverages **Google Gemini Flash** for reasoning and a **HuggingFace model specific to Italian** (`nickprock/sentence-bert-base-italian-xxl-uncased`) for semantic search (embeddings).
* **Efficiency:** The entire agent pipeline (models, vector store) is loaded only once thanks to Streamlit's cache (`@st.cache_resource`), making subsequent analyses instantaneous.

## Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Orchestration:** [LangChain](https://www.langchain.com/) (Agents, Tools, Prompts)
* **LLM Model:** [Google Gemini Flash](https://ai.google.com/gemini/) (via `langchain-google-genai`)
* **Embeddings:** [HuggingFace](https://huggingface.co/) (`sentence-transformers`)
* **Vector Store:** [ChromaDB](https://www.trychroma.com/) (in-memory)

---

## Setup and Launch

There are two main ways to run this project: locally or via Streamlit Community Cloud.

### 1. Running on Streamlit Community Cloud (Recommended)

This script is already optimised for deployment on Streamlit Cloud.

1.  Ensure your GitHub repository contains **3 essential files**:
    * `agent_app.py` (the main Python script)
    * `regole.txt` (the text file with your knowledge base)
    * `requirements.txt` (see below)
2.  Create the `requirements.txt` file and add the following dependencies:
    ```txt
    streamlit
    langchain-community
    langchain-text-splitters
    langchain-google-genai
    langchain-huggingface
    langchain-classic
    chroma-db
    sentence-transformers
    ```
3.  Connect your GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io/).
4.  In your app's **Settings** on Streamlit Cloud, go to the **Secrets** section and add your Google API key:
    ```toml
    GOOGLE_API_KEY = "your_google_api_key_starts_with_AIza..."
    ```
5.  Deploy the app. It will start automatically.

### 2. Running Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
    cd your-repo
    ```
2.  **Create the `regole.txt` file:** Ensure the `regole.txt` file (with your knowledge base) is present in the same folder.
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install the dependencies:** Create a `requirements.txt` file as shown above and run:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure the Secrets:** Streamlit looks for secrets in a `.streamlit` folder inside the project directory.
    * Create the folder: `mkdir .streamlit`
    * Create the secrets file: `nano .streamlit/secrets.toml`
    * Paste your key inside the file:
        ```toml
        GOOGLE_API_KEY = "your_google_api_key_starts_with_AIza..."
        ```
6.  **Run the app:**
    ```bash
    streamlit run agent_app.py
    ```

---

## How the Agent Works

The app's intelligence lies in the `get_agent_executor()` function, which is cached.

1.  **Loading (Setup):** On the first run, the app loads the Gemini LLM, the HuggingFace embedding model, and the `regole.txt` file.
2.  **Indexing (RAG):** The `regole.txt` file is split, transformed into vectors (embeddings), and indexed into an in-memory Chroma vector database.
3.  **Tool:** A single tool, `cerca_regole_semplificazione` (search simplification rules), is defined. This tool allows the agent to semantically "search" for the most relevant rules within the Chroma database.
4.  **Prompt (Instructions):** The agent receives a highly detailed prompt (`AGENT_PROMPT_TEMPLATE`) that instructs it to behave like a consultant. It is told to:
    * Identify problem **categories** (a top-down approach).
    * Use the `cerca_regole_semplificazione` tool to find the **official rule** corresponding to that category.
    * Provide a final response structured in Markdown, grouping text examples under the identified categories.
5.  **Execution (Runtime):** When the user clicks "Analizza Testo" (Analyse Text), the ReAct agent receives the input and executes a "Thought -> Action -> Observation" cycle to build its final answer.

## Contributing

Contributions are welcome! Please open an Issue to discuss what you would like to change, or submit a Pull Request directly.
