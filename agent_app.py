# -*- coding: utf-8 -*-

import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter # Using this as per your code
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor

# CACHING FUNCTION FOR AGENT SETUP
@st.cache_resource
def get_agent_executor():
    # Loading API Key
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("ERROR: GOOGLE_API_KEY not found.")
        st.info("Add your API key in the 'Secrets' of the Streamlit app settings.")
        st.stop() # Stop execution if the key is missing

    # LLM Initialization
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=GOOGLE_API_KEY,
                                 temperature=0.1,
                                 max_output_tokens=4000,
                                 convert_system_message_to_human=True)

    # Document Loading
    try:
        loader = TextLoader("regole.txt")
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading 'regole.txt' file: {e}")
        st.stop()

    # Splitter and Embeddings

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Load the Italian embedding model
    model_name = "nickprock/sentence-bert-base-italian-xxl-uncased"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Create the vector store
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Tool Definition
    @tool
    def search_simplification_rules(topic):
        """
        Searches the knowledge base for rules and advice
        for a specific linguistic simplification topic.
        """
        docs_tool = retriever.invoke(topic)
        return "\n\n".join(doc.page_content for doc in docs_tool)

    tools = [search_simplification_rules]

    # Prompt and Agent Creation
    AGENT_PROMPT_TEMPLATE = """
    Sei un consulente esperto in semplificazione del linguaggio amministrativo.
    Analizza il testo fornito e fornisci un'analisi sintetica, ragionata e strutturata.

    OBIETTIVO:
    - Identificare le principali CATEGORIE di problemi linguistici (livello alto).
    - Per ciascuna categoria, mostrare esempi specifici tratti dal testo e fornire la regola o il consiglio corrispondente.
    - Non elencare ogni singola criticità minore. Raggruppa i casi simili sotto la stessa categoria.
    - L’obiettivo è una risposta compatta e utile alla revisione del testo.

    STRUMENTI DISPONIBILI:
    {tools}
    (Puoi usare uno dei seguenti strumenti: {tool_names})

    FORMATO PER CHIAMARE GLI STRUMENTI:
    Step: [Breve motivazione, max 10 parole]
    Action: [nome_strumento]
    Action Input: ["query sintetica, es. regola per frasi lunghe"]
    Observation: [attendi il risultato dallo strumento]

    LINEA GUIDA (approccio top-down):
    1. Leggi il testo e individua le principali categorie di criticità linguistiche generali.
    2. Per ogni categoria, usa lo strumento `cerca_regole_semplificazione` per trovare la regola o il consiglio corrispondente.
    3. Collega alla categoria solo alcuni esempi rappresentativi dal testo (non tutti).
    4. Alla fine, sintetizza le categorie e i consigli in una risposta finale chiara e breve.

    OUTPUT FINALE (obbligatorio):
    Final Answer:
    Ecco un riepilogo delle principali aree di miglioramento del testo:

    **Categoria 1: [Nome del problema generale]**
    * **Descrizione:** [Spiega la natura del problema e il principio di semplificazione]
    * **Esempi nel testo:** [2-3 esempi rappresentativi]
    * **Regola/Consiglio:** [Sintesi della regola o del consiglio pratico]

    **Categoria 2: [Altro problema generale]**
    ...

    IMPORTANTE:
    - Non elencare ogni singolo errore.
    - Raggruppa le osservazioni in poche categorie significative.
    - La risposta finale deve iniziare con "Final Answer:" e terminare subito dopo l'elenco.
    - Nessun saluto o testo aggiuntivo dopo la risposta finale.

    Inizia ora!

    Testo da analizzare:
    {input}
    Traccia delle azioni e osservazioni passate:
    {agent_scratchpad}
    """

    agent_prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agent_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=10
    )

    return agent_executor

# STREAMLIT USER INTERFACE (UI)

st.title("Simplification Agent")
st.markdown("This agent analyzes administrative text and identifies areas for improvement, suggesting relevant simplification rules.")

# Try to load the agent (using the cache)
# This will show st.info/st.success messages the first time
try:
    agent_executor = get_agent_executor()
except Exception as e:
    st.error(f"Critical error during agent initialization: {e}")
    st.stop()

# Text area for user input
testo_da_analizzare = st.text_area("Paste the text to analyze here:", height=250,
                                  placeholder="E.g., 'You are hereby notified that the meeting will take place on this day...'")

# Button to start processing
if st.button("Analyze Text"):
    if testo_da_analizzare:
        # Show a spinner while the agent works
        with st.spinner("The agent is thinking... (this could take up to 30 seconds)"):
            try:
                # Agent Invocation
                result = agent_executor.invoke({"input": testo_da_analizzare})

                st.markdown(result['output'])

            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
    else:
        st.warning("Please enter some text in the box above.")
