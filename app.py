"""
app.py
------
Main Streamlit app for the Superhero Character Chatbot.
Uses RAG (LangChain + FAISS) to retrieve hero dialogues
and Groq LLM to respond in character.

Run with: streamlit run app.py
"""

import os
import yaml
import streamlit as st

from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_FILE = "config.yaml"
DIALOGUES_FOLDER = "dialogues"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "**********************************")

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

SUPERHEROES = config["LIST_OF_SUPERHEROES"]
PERSONALITIES = config["SUPERHERO_PERSONALITIES"]

# ─── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hero Chat",
    page_icon="🦸",
    layout="wide"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0d0d0d;
        color: #f0f0f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 2px solid #e94560;
    }

    /* Title */
    h1 {
        color: #e94560;
        text-align: center;
        font-family: 'Courier New', monospace;
        letter-spacing: 3px;
    }

    /* Hero message bubble */
    .hero-bubble {
        background-color: #16213e;
        border-left: 4px solid #e94560;
        padding: 12px 16px;
        border-radius: 0px 12px 12px 12px;
        margin: 8px 0;
        color: #f0f0f0;
        max-width: 75%;
    }

    /* User message bubble */
    .user-bubble {
        background-color: #1a1a2e;
        border-right: 4px solid #0f3460;
        padding: 12px 16px;
        border-radius: 12px 0px 12px 12px;
        margin: 8px 0 8px auto;
        color: #f0f0f0;
        max-width: 75%;
        text-align: right;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1a1a2e;
        color: #f0f0f0;
        border: 1px solid #e94560;
        border-radius: 8px;
    }

    /* Button */
    .stButton > button {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #c73652;
    }

    /* Selectbox */
    .stSelectbox > div {
        background-color: #16213e;
        color: #f0f0f0;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #e94560 !important;
    }

    /* Divider */
    hr {
        border-color: #e94560;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DIALOGUE TEXT ────────────────────────────────────────────────────────

def load_hero_dialogues(superhero):
    """
    Load the extracted dialogue .txt file for the selected hero.
    Returns raw text or None if file doesn't exist.
    """
    hero_folder = os.path.join(DIALOGUES_FOLDER, superhero.replace(" ", "_"))
    dialogue_file = os.path.join(hero_folder, "dialogues.txt")

    if not os.path.exists(dialogue_file):
        return None

    with open(dialogue_file, "r", encoding="utf-8") as f:
        return f.read()


# ─── BUILD VECTOR STORE ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_vectorstore(superhero):
    """
    Takes the hero's dialogue text, splits into chunks,
    embeds them, and stores in FAISS for retrieval.
    Cached so it doesn't rebuild every time.
    """
    text = load_hero_dialogues(superhero)

    if not text:
        return None

    # Split dialogue into small chunks
    splitter = CharacterTextSplitter(
        separator="---dialogue-separator---",
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    if not chunks:
        return None

    # Free, local embedding model — no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build FAISS vector store from chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ─── BUILD RAG CHAIN ───────────────────────────────────────────────────────────

def build_chain(superhero, vectorstore):
    """
    Build a simple RAG chain using the new LangChain API:
    - Retriever fetches relevant dialogue chunks
    - LLM responds in character using personality prompt
    - Streamlit maintains conversation history in session state
    - Chat history is passed to LLM for memory
    """
    personality = PERSONALITIES.get(superhero, f"You are {superhero}. Stay in character.")

    # System prompt — this is what makes the hero SOUND like the hero
    system_prompt = f"""
{personality}

You have been given some of your actual movie dialogue as context below.
Use it to understand how you speak, your tone, vocabulary, and personality.
Always respond AS the character. Never say you are an AI. Never break character.
Reference your dialogue examples when appropriate to stay authentic.
Keep responses concise but in character."""

    # LLM — Groq is free and fast
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.8  # slightly creative so it feels natural
    )

    # Retriever — fetch top 4 most relevant dialogue chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Return a simple function that uses the retriever and LLM
    def chain_func(data):
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")
        
        question = data.get("question", "")
        chat_history = data.get("chat_history", [])  # Get chat history from input
        
        if not question:
            raise ValueError("Question is required in input dict")
        
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format chat history for context
        chat_history_str = ""
        if chat_history:
            chat_history_str = "\nPrevious conversation:\n"
            for role, msg in chat_history[-6:]:  # Include last 6 messages for context
                chat_history_str += f"{role.capitalize()}: {msg}\n"
        
        # Build the system prompt with context
        system_content = f"""{system_prompt}

Your dialogue examples:
{context}{chat_history_str}"""
        
        # Build messages with history
        messages = [SystemMessage(content=system_content)]
        
        # Add chat history to messages for LLM context
        for role, msg in chat_history[-4:]:  # Last 4 messages for conversation flow
            if role == "user":
                messages.append(HumanMessage(content=msg))
            else:
                messages.append(SystemMessage(content=f"You previously said: {msg}"))
        
        # Add current question
        messages.append(HumanMessage(content=question))
        
        # Get response from LLM
        response = llm.invoke(messages)
        return {"answer": response.content}
    
    return chain_func


# ─── SESSION STATE INIT ────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_hero" not in st.session_state:
    st.session_state.current_hero = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🦸 Hero Selector")
    st.markdown("---")

    selected_hero = st.selectbox(
        "Choose your hero:",
        SUPERHEROES,
        index=0
    )

    st.markdown("---")
    st.markdown(f"**Chatting with:**")
    st.markdown(f"### {selected_hero}")
    st.markdown("---")

    if st.button("🔄 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.chain = None
        st.rerun()

    st.markdown("---")
    st.markdown("*Powered by LangChain + FAISS + Groq*")


# ─── MAIN AREA ─────────────────────────────────────────────────────────────────

st.markdown("# 🦸 HERO CHAT")
st.markdown("##### *Talk to your favorite superhero — they'll respond in character!*")
st.markdown("---")

# If hero changed, reset everything
if selected_hero != st.session_state.current_hero:
    st.session_state.current_hero = selected_hero
    st.session_state.chat_history = []
    st.session_state.chain = None
    st.session_state.vectorstore = None

# Load vectorstore and chain for selected hero
if st.session_state.chain is None:
    with st.spinner(f"Loading {selected_hero}'s dialogue data..."):
        vs = build_vectorstore(selected_hero)

        if vs is None:
            st.error(
                f"❌ No dialogue file found for **{selected_hero}**.\n\n"
                f"Please run `python extract_dialogues.py` first, "
                f"and make sure PDFs are in the `pdfs/` folder."
            )
            st.stop()

        st.session_state.vectorstore = vs
        st.session_state.chain = build_chain(selected_hero, vs)

# Show chat history
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {message}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="hero-bubble">🦸 <b>{selected_hero}:</b> {message}</div>',
                unsafe_allow_html=True
            )

# Input area
st.markdown("---")
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your message:",
        placeholder=f"Say something to {selected_hero}...",
        label_visibility="collapsed",
        key="user_input_field"
    )

with col2:
    send = st.button("Send 🚀")

# Handle sending message
if send and user_input.strip():
    with st.spinner(f"{selected_hero} is thinking..."):
        try:
            # Pass chat history to chain so it remembers context
            result = st.session_state.chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            response = result["answer"].strip()
        except Exception as e:
            response = f"[Error getting response: {e}]"

    # Save to chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("hero", response))

    st.rerun()

