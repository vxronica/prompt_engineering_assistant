# Prompt Engineering Chatbot

An educational Q&A chatbot that teaches prompt engineering techniques and strategies for reducing hallucinations in AI-generated responses.

This chatbot is designed for college students, AI beginners, and everyday users who want to learn how to write clearer prompts and get more reliable AI outputs.

---

## Features

- Retrieval-Augmented Generation (RAG) chatbot
- Semantic search using Sentence Transformers
- FLAN-T5 response generation
- Knowledge-base grounded answers
- Hallucination-aware fallback responses
- Responsive chatbot UI for desktop and mobile
- AI/user avatars
- Typing indicator animation
- Simulated thinking delay

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
git clone https://github.com/vxronica/prompt_engineering_assistant.git
cd prompt_engineering_assistant
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

1. The user enters a question in the chatbot UI.
2. The question is cleaned and corrected for small typos.
3. The question is converted into an embedding using Sentence Transformers.
4. The system compares the question embedding to the knowledge base embeddings.
5. The top-k most relevant knowledge base chunks are retrieved.
6. The retrieved chunks are inserted into a prompt for FLAN-T5.
7. FLAN-T5 generates a concise answer using only the retrieved context.
8. If retrieval confidence is low or the answer goes off-topic, the chatbot returns a safe fallback response.
---

## Example Questions

- What is prompt engineering?
- How can I write better prompts?
- What is a token?
- What causes AI hallucinations?
- What is RAG?
- What is the difference between zero-shot and few-shot prompting?
- Which is better, chain-of-thought or few-shot prompting?
- What is temperature?

---

