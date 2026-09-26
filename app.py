import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from nlp.scope_checker import ScopeChecker
from rag.retriever import Retriever

st.set_page_config(page_title="NLP Knowledge Assistant", page_icon="🧠", layout="centered")

@st.cache_resource

def load_components():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    scope_checker = ScopeChecker(model)
    retriever = Retriever(model)
    return model, scope_checker, retriever

st.title("🧠 NLP Knowledge Assistant")
st.write("Ask questions about Natural Language Processing, from basic concepts to modern NLP, transformers, LLMs, embeddings, vector databases, and RAG.")

with st.expander("What can I ask?"):
    st.markdown("""
    - What is NLP?
    - What are the types of NLP?
    - What is tokenization?
    - What is TF-IDF?
    - Explain Word2Vec.
    - What is sentiment analysis?
    - What is a transformer?
    - Explain self-attention.
    - What is BERT vs GPT?
    - What is an embedding?
    - What is ChromaDB?
    - What is RAG?
    """)

try:
    _, scope_checker, retriever = load_components()
except Exception as e:
    st.error("The NLP components could not be loaded.")
    st.code(str(e))
    st.info("If this is the first run, wait for the embedding model to finish downloading and then refresh the page.")
    st.stop()

query = st.text_area("Ask your NLP question:", placeholder="Example: Explain how self-attention works in transformers")

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    in_scope, score = scope_checker.check(query.strip())

    if not in_scope:
        st.warning("I’m an NLP-focused assistant. Please ask about NLP concepts, techniques, models, algorithms, applications, embeddings, LLMs, or RAG.")
        st.caption(f"NLP relevance score: {score:.2f}")
        st.stop()

    documents = retriever.search(query.strip(), k=4)
    context = "\n\n---\n\n".join(documents)

    answer = None
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    try:
        from ollama import chat
        response = chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an NLP Knowledge Assistant. Answer only using the supplied NLP knowledge context. Explain clearly for a student. If the context does not contain enough information, say so instead of inventing facts."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query.strip()}"
                }
            ]
        )
        answer = response["message"]["content"]
    except Exception as e:
        st.info("Ollama is not available, so I’m showing the retrieved NLP knowledge directly. You can install/start Ollama and use the LLM-generated explanation later.")
        st.caption(f"Ollama detail: {e}")
        answer = context

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved NLP knowledge"):
        st.write(context)
