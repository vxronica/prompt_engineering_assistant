# Prompt Engineering Chatbot

An interactive chatbot and tutorial system that teaches prompt engineering concepts using a Retrieval-Augmented Generation (RAG) pipeline.

This project combines a knowledge-based retrieval system with a language model to provide accurate, grounded responses while reducing hallucinations.

---

## Features

- Built-in tutorial panel for prompt engineering concepts
- Chatbot powered by RAG
- Hallucination-aware fallback responses
- Chat-style HTML/CSS frontend
- Semantic search using Sentence Transformers
- Lightweight generation using FLAN-T5

---

## Tech Stack

- Python
- Flask
- HTML
- CSS
- JavaScript
- Sentence Transformers
- Hugging Face Transformers
- PyTorch

---

## Project Structure

```text
prompt_engineering_assistant/
├── app.py
├── chatbot.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/prompt-engineering-chatbot.git
cd prompt-engineering-chatbot
```

### 2. Create a virtual environment

Mac / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

Start the Flask server:

```bash
python app.py
```

Open your browser and go to:

```text
http://127.0.0.1:5050
```

---

## How It Works

1. The user enters a question in the UI.
2. The question is cleaned and processed.
3. The question is embedded using Sentence Transformers.
4. The system retrieves the most relevant knowledge base entries.
5. The retrieved context is passed into FLAN-T5.
6. The model generates a grounded response.
7. If confidence is low, the chatbot returns a fallback response.

---

## Example Questions

- What is prompt engineering?
- What is a token?
- How can I reduce hallucinations?
- What is RAG?

---

