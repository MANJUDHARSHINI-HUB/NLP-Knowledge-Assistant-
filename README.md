# NLP Knowledge Assistant

A beginner-friendly NLP-focused RAG application built with Python, Streamlit, Sentence Transformers, ChromaDB, and optional Ollama.

## Run on Windows

Open CMD and run:

```cmd
cd C:\Users\acer\Downloads\NLP-Knowledge-Assistant
"C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt
"C:\Users\acer\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py
```

If Ollama is installed and the model is available, the app can use `llama3.2:3b` for the final explanation. If Ollama is unavailable, the app still shows the retrieved NLP knowledge instead of crashing.

## Example questions

- What is NLP?
- What are the types of NLP?
- What is tokenization?
- Explain TF-IDF.
- What is BERT?
- What is self-attention?
- What is RAG?
- What is ChromaDB?

The app is intentionally NLP-focused and rejects unrelated questions.

Live Link : https://3txnur2tqgre46dqgqogat.streamlit.app/
