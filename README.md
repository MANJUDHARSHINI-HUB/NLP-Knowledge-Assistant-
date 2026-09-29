# 🧠 NLP Knowledge Assistant

An AI-powered **NLP Knowledge Assistant** that helps users learn and understand **Natural Language Processing (NLP)** concepts through question answering and topic-based learning.

The application uses **semantic search, Sentence Transformers, ChromaDB, RAG, and a Groq LLM** to retrieve relevant NLP knowledge and generate clear explanations.

---

## 📌 Project Overview

The NLP Knowledge Assistant is designed specifically for **NLP-related questions**.

Users can:

* Ask questions about NLP
* Learn a complete NLP topic using the **Learn an NLP Topic** feature
* Get answers based on a predefined NLP knowledge base
* Search information using semantic meaning instead of exact keywords
* View the knowledge retrieved by the system
* Use suggested questions for quick learning

The system also checks whether a question is related to NLP before generating an answer.

---

## 🎯 Main Objective

The main objective of this project is to create a focused AI assistant that can:

1. Identify whether a question is related to NLP.
2. Find the most relevant NLP information from a knowledge base.
3. Use the retrieved information as context.
4. Generate a clear answer using an LLM.
5. Help beginners learn NLP topics in a structured way.

---

## ✨ Features

### 💬 Ask NLP Questions

Users can ask questions such as:

* What is NLP?
* What is tokenization?
* What is TF-IDF?
* How does sentiment analysis work?
* What is BERT?
* What are embeddings?
* What is RAG?
* What is self-attention?

---

### 📖 Learn an NLP Topic

The application provides a **Learn an NLP Topic** feature.

The user selects a topic such as:

* NLP
* Tokenization
* TF-IDF
* Word2Vec
* Sentiment Analysis
* BERT
* GPT
* Self-Attention
* Embeddings
* ChromaDB
* RAG
* Transformers

When the user clicks **Learn This Topic**, the application automatically creates a question such as:

```text
Explain BERT
```

The system then retrieves relevant knowledge about BERT and asks the LLM to create a beginner-friendly lesson.

---

### 🔎 NLP Scope Checking

The application checks whether the user's question is related to NLP.

For example:

```text
What is tokenization?
```

is considered an NLP-related question.

But:

```text
What is the weather today?
```

is outside the NLP scope.

This prevents the assistant from behaving like a general-purpose chatbot.

---

### 🧠 Semantic Search

The application uses semantic search instead of only matching exact keywords.

The Sentence Transformer converts text into numerical **embeddings** that represent meaning.

ChromaDB then compares the embeddings and retrieves information that is semantically similar to the user's question.

For example:

```text
User question:
What information does RAG retrieve?

Knowledge:
RAG retrieves relevant information from a knowledge base
before generating an answer.
```

Even though the wording is different, the meanings are related.

---

### 📚 RAG-Based Question Answering

The project uses **Retrieval-Augmented Generation (RAG)**.

The basic process is:

```text
User Question
      ↓
Scope Checking
      ↓
Create Embedding
      ↓
Semantic Search
      ↓
ChromaDB
      ↓
Retrieve Relevant Knowledge
      ↓
Add Knowledge to Prompt
      ↓
Groq LLM
      ↓
Generate Answer
```

RAG helps the LLM generate answers using information retrieved from the project's NLP knowledge base.

---

## 🏗️ Project Architecture

```text
                    USER
                      │
                      ▼
               Streamlit UI
                      │
                      ▼
                User Question
                      │
                      ▼
               Scope Checker
                  /       \
                NO         YES
                │           │
                ▼           ▼
          Out-of-scope   Sentence
             Response     Transformer
                              │
                              ▼
                         ChromaDB
                              │
                              ▼
                    Relevant NLP Knowledge
                              │
                              ▼
                             RAG
                              │
                              ▼
                         Groq LLM
                              │
                              ▼
                      Generated Answer
                              │
                              ▼
                            USER
```

---

## 🔄 How the Project Works

### Step 1 — User enters a question

The user enters a question through the Streamlit interface.

Example:

```text
What is BERT?
```

---

### Step 2 — Scope checking

The **Scope Checker** determines whether the question is related to NLP.

It uses a Sentence Transformer to compare the meaning of the question with predefined NLP-related examples.

If the question is outside the NLP domain, the application gives an out-of-scope response.

---

### Step 3 — Create an embedding

The Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

converts the question into an embedding.

An embedding is a numerical representation of the meaning of the text.

---

### Step 4 — Semantic search

The Retriever sends the question embedding to ChromaDB.

ChromaDB searches the stored NLP knowledge using semantic similarity.

The application retrieves the most relevant sections from the knowledge base.

---

### Step 5 — Create the RAG context

The retrieved information is combined into a context.

This context is added to the prompt sent to the LLM.

For example:

```text
Retrieved NLP Knowledge:
BERT is a Transformer-based language model...
BERT uses bidirectional context...
BERT can be used for NLP tasks...

User Question:
What is BERT?
```

---

### Step 6 — Groq LLM generates the answer

The application sends the prompt to the Groq API.

The LLM uses:

* The user's question
* The retrieved NLP knowledge
* The instructions in the prompt

to generate the final answer.

---

### Step 7 — Display the answer

The generated answer is displayed in the Streamlit interface.

The application can also show the retrieved knowledge used for the answer.

---

# 📖 How the Learn Feature Works

The Learn feature uses the same RAG pipeline but provides additional instructions to the LLM.

For example, the user selects:

```text
BERT
```

The application automatically creates:

```text
Explain BERT
```

The Retriever searches ChromaDB for relevant BERT information.

The retrieved information is added to a learning prompt.

The LLM is instructed to explain the topic in a beginner-friendly way.

The learning response can cover areas such as:

1. What is it?
2. Why is it used?
3. How does it work?
4. Main concepts
5. Types or variations
6. Simple example
7. Applications
8. Advantages
9. Limitations
10. Quick recap

So the Learn feature is essentially:

```text
Selected Topic
      ↓
"Explain [Topic]"
      ↓
Sentence Transformer
      ↓
ChromaDB Semantic Search
      ↓
Relevant Knowledge
      ↓
Learning Prompt
      ↓
Groq LLM
      ↓
Beginner-Friendly Lesson
```

---

# 📂 Project Structure

```text
NLP-Knowledge-Assistant/
│
├── app.py
│
├── data/
│   └── nlp_knowledge.md
│
├── nlp/
│   └── scope_checker.py
│
├── rag/
│   └── retriever.py
│
├── requirements.txt
│
├── README.md
│
└── chroma_db/
```

---

## 📄 File Explanation

### `app.py`

This is the main application file.

It handles:

* Streamlit UI
* User input
* Suggested questions
* Learn topic feature
* Scope checking
* Retrieval
* RAG prompt creation
* Groq API call
* Displaying the final answer
* Recent questions

---

### `data/nlp_knowledge.md`

This is the project's NLP knowledge base.

It contains information about different NLP concepts.

The Retriever reads this file and divides it into sections before storing the information in ChromaDB.

---

### `nlp/scope_checker.py`

This file checks whether a user's question is related to NLP.

Its main purpose is:

```text
"Should this question be answered by my NLP assistant?"
```

---

### `rag/retriever.py`

This file handles retrieval.

Its main purpose is:

```text
"What NLP information should I use to answer this question?"
```

It:

1. Reads the NLP knowledge base.
2. Creates embeddings for the knowledge.
3. Stores them in ChromaDB.
4. Converts the user's question into an embedding.
5. Searches ChromaDB.
6. Returns the most relevant knowledge.

---

## 🛠️ Technologies Used

| Technology            | Purpose                               |
| --------------------- | ------------------------------------- |
| Python                | Main programming language             |
| Streamlit             | User interface                        |
| Sentence Transformers | Creates text embeddings               |
| all-MiniLM-L6-v2      | Embedding model                       |
| ChromaDB              | Vector database and semantic search   |
| RAG                   | Retrieves knowledge before generation |
| Groq                  | LLM API                               |
| Markdown              | NLP knowledge base                    |

---

## 🧩 Why Each Technology Is Used

### Python

Used to build the application logic and connect all the components.

### Streamlit

Used to create the web interface without building a separate frontend.

### Sentence Transformers

Used to convert text into embeddings.

### ChromaDB

Used to store embeddings and perform semantic similarity search.

### RAG

Used to retrieve relevant information and provide it as context to the LLM.

### Groq

Used to generate the final natural-language response.

---

## 🧠 Important Concepts

### Embedding

An embedding is a numerical representation of text meaning.

```text
Text
 ↓
Sentence Transformer
 ↓
Embedding
```

---

### Semantic Search

Semantic search finds information based on **meaning**, rather than only matching exact words.

---

### Vector Database

A vector database stores embeddings and allows similarity searches.

In this project, **ChromaDB** acts as the vector database.

---

### RAG

RAG stands for:

**Retrieval-Augmented Generation**

It combines:

```text
Retrieval + Generation
```

The system first retrieves relevant knowledge and then gives that knowledge to the LLM to generate the answer.

---

## 🚀 Installation

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
```

Move into the project folder:

```bash
cd NLP-Knowledge-Assistant
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Groq API Key

The Groq API key should **not** be written directly inside the Python code or uploaded to GitHub.

For Streamlit deployment, add the key through **Streamlit Secrets**.

Example:

```toml
GROQ_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the Application

Run the application using:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

---

## 🌐 Deployment

The project can be deployed using **Streamlit Community Cloud**.

The GitHub repository should contain the application files and the required dependencies.

The Groq API key should be added securely using Streamlit Secrets.

---

## 🔐 Security

The Groq API key should never be committed to GitHub.

Use environment variables or Streamlit Secrets instead.

Do not write:

```python
GROQ_API_KEY = "actual-secret-key"
```

inside the source code.

---

## 💡 Example Questions

```text
What is NLP?
```

```text
What is tokenization?
```

```text
What is TF-IDF?
```

```text
How does sentiment analysis work?
```

```text
What is BERT?
```

```text
What are embeddings?
```

```text
What is self-attention?
```

```text
What is RAG?
```

```text
What is ChromaDB?
```

```text
How do Transformers work?
```

---

## 🎓 Project Learning Outcomes

Through this project, I learned about:

* Natural Language Processing
* Text embeddings
* Sentence Transformers
* Semantic search
* Vector databases
* ChromaDB
* Retrieval-Augmented Generation
* LLM integration
* Prompt engineering
* Streamlit application development
* API integration
* Knowledge-based question answering

---

## 🔮 Future Improvements

Possible future improvements include:

* Adding more NLP knowledge
* Supporting more advanced NLP topics
* Adding conversation history
* Adding source citations
* Improving retrieval accuracy
* Adding document upload functionality
* Adding multiple knowledge bases
* Adding multilingual NLP support

---

## 👩‍💻 Project Summary

**NLP Knowledge Assistant** is a domain-specific AI assistant that combines **Sentence Transformers, ChromaDB, RAG, and Groq** to answer NLP-related questions.

The application first checks whether a question belongs to the NLP domain, retrieves relevant information using semantic search, and then uses an LLM to generate a clear response.

The project demonstrates how **semantic search + retrieval + LLM generation** can be combined to build a focused AI knowledge assistant.

---

## 📌 Key Pipeline

```text
User Question
      ↓
Streamlit
      ↓
Scope Checker
      ↓
Sentence Transformer
      ↓
ChromaDB
      ↓
Relevant NLP Knowledge
      ↓
RAG
      ↓
Groq LLM
      ↓
Final Answer
```

**Built with Python, Streamlit, Sentence Transformers, ChromaDB, RAG, and Groq.**
