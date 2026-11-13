# Conversational PDF Q&A Chatbot

A Streamlit application for asking grounded questions about PDF and text documents. It combines local semantic search with a conversational LLM so follow-up questions can use recent chat history without inventing facts outside the retrieved document excerpts.

## Features

- Load PDF and UTF-8 TXT documents.
- Split long documents into overlapping chunks.
- Generate embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve relevant passages with Chroma.
- Keep a bounded conversation history for follow-up questions.
- Choose Ollama for local inference or OpenAI for hosted inference.
- Display source file names and PDF page numbers with answers.

## How it works

```text
PDF/TXT upload
  -> document loading and chunking
  -> local embeddings and Chroma vector store
  -> retrieve relevant chunks for the question
  -> answer prompt + recent conversation
  -> source-grounded response
```

The answer prompt restricts responses to retrieved context and instructs the model to say **"I don't know"** when the documents do not provide an answer.

## Requirements

- Python 3.10 or later
- For local inference: [Ollama](https://ollama.com/) and a downloaded chat model
- For hosted inference: an OpenAI API key

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Choose a model provider in the app's sidebar.

### Ollama

Pull the default model before starting the app:

```powershell
ollama pull llama3.1
streamlit run app.py
```

Set `OLLAMA_MODEL` in `.env` to use a different locally available model.

### OpenAI

Set `OPENAI_API_KEY` in `.env` and optionally change `OPENAI_MODEL`:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
```

Then start the app:

```powershell
streamlit run app.py
```

## Usage

1. Upload one or more PDF or TXT files in the sidebar.
2. Select **Build knowledge base** and wait for indexing to finish.
3. Ask a factual question, then try a follow-up such as "Can you explain that further?"
4. Use **Clear chat history** to start a new conversation without rebuilding the index.

`data/sample.txt` is a fictional example document you can upload to try the workflow.

## Project structure

```text
.
├── app.py
├── data/
│   └── sample.txt
├── src/
│   └── rag_pipeline.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Privacy

Document parsing, embeddings, and vector search run locally. With Ollama, inference also runs locally. When OpenAI is selected, the question, recent conversation, and retrieved excerpts are sent to the OpenAI API. Do not upload documents you are not authorized to process, and keep `.env` private.
