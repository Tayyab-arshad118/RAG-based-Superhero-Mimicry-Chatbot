# 🦸 Hero Chat — Superhero RAG Chatbot

Talk to your favorite superhero. They respond in character using their actual movie dialogues.

---

## 📁 Folder Structure

```
superhero-chatbot/
│
├── pdfs/                  ← Put ALL your movie script PDFs here
├── dialogues/             ← Auto-created after extraction
├── config.yaml            ← Hero names, synonyms, movie list, personalities
├── extract_dialogues.py   ← Step 1: Extract hero dialogues from PDFs
├── app.py                 ← Step 2: Run the chatbot
├── requirements.txt
└── README.md
```

---

## 🚀 Setup (Step by Step)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your PDFs
Put all movie script PDFs inside the `pdfs/` folder.
Make sure the filenames match what's in `config.yaml`.

### 3. Add your Groq API Key
Get a FREE key from: https://console.groq.com

Open `app.py` and replace:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```

Or set it as an environment variable:
```bash
export GROQ_API_KEY="your_key_here"
```

### 4. Extract dialogues
```bash
python extract_dialogues.py
```
This reads the PDFs and saves only each hero's dialogues into the `dialogues/` folder.
Run this ONCE. Re-run only if you add new PDFs.

### 5. Run the chatbot
```bash
streamlit run app.py
```

---

## 🎭 How It Works

```
PDFs in pdfs/ folder
        ↓
extract_dialogues.py
  - Finds hero name (+ synonyms) in script
  - Extracts only their dialogue blocks
  - Saves to dialogues/HeroName/dialogues.txt
        ↓
app.py loads dialogues
  - Splits into chunks (CharacterTextSplitter)
  - Embeds with HuggingFace (free, local)
  - Stores in FAISS vector store
        ↓
User asks a question
  - Retriever fetches top 4 relevant dialogue chunks
  - System prompt loads hero personality
  - Groq LLM responds IN CHARACTER
  - Memory keeps conversation history
```

---

## 🔧 Customization

- Add more heroes → Edit `config.yaml`
- Change personality → Edit `SUPERHERO_PERSONALITIES` in `config.yaml`
- Add more PDFs → Drop in `pdfs/` folder, re-run extraction
- Change LLM model → Edit `model_name` in `app.py`

---

## ⚠️ Notes

- First run downloads the embedding model (~90MB, one time only)
- Groq API is FREE with generous limits
- Works best with proper movie script PDFs (not scanned images)
