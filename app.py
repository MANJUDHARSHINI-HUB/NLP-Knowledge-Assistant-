```python
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

from nlp.scope_checker import ScopeChecker
from rag.retriever import Retriever


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NLP Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# TITLE AND DESCRIPTION
# =========================================================

st.title("🧠 NLP Knowledge Assistant")

st.write(
    "Ask questions about Natural Language Processing (NLP), "
    "including concepts, techniques, algorithms, models, "
    "applications, Transformers, LLMs, embeddings, and RAG."
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# =========================================================
# LOAD PROJECT COMPONENTS
# =========================================================

@st.cache_resource
def load_components():

    # Load Sentence Transformer
    model = load_embedding_model()

    # NLP scope checker
    scope_checker = ScopeChecker(
        model=model,
        threshold=0.38
    )

    # ChromaDB retriever
    # Retriever expects "embedding_model"
    retriever = Retriever(
        embedding_model=model
    )

    return model, scope_checker, retriever


model, scope_checker, retriever = load_components()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 What can I ask?")

    st.write(
        "This assistant is designed specifically "
        "for NLP-related questions."
    )

    st.write("Examples:")

    st.write("• What is NLP?")
    st.write("• What are the types of NLP?")
    st.write("• What is tokenization?")
    st.write("• Explain TF-IDF")
    st.write("• What is Word2Vec?")
    st.write("• What is sentiment analysis?")
    st.write("• Explain BERT")
    st.write("• BERT vs GPT")
    st.write("• What is self-attention?")
    st.write("• What are embeddings?")
    st.write("• What is ChromaDB?")
    st.write("• What is RAG?")
    st.write("• How does a Transformer work?")

    st.divider()

    st.info(
        "Questions unrelated to NLP are outside "
        "the scope of this assistant."
    )


# =========================================================
# USER INPUT
# =========================================================

query = st.text_input(
    "💬 Ask your NLP question:",
    placeholder="Example: What is tokenization?"
)


# =========================================================
# ASK BUTTON
# =========================================================

if st.button("Ask", type="primary"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # =================================================
        # STEP 1: NLP SCOPE CHECK
        # =================================================

        with st.spinner(
            "Checking whether your question is NLP-related..."
        ):

            in_scope, score = scope_checker.check(
                query
            )


        # =================================================
        # OUT-OF-SCOPE QUESTION
        # =================================================

        if not in_scope:

            st.warning(
                "Sorry, I’m an NLP-focused assistant. "
                "I can answer questions about NLP concepts, "
                "techniques, models, algorithms, and applications."
            )

            st.caption(
                f"NLP relevance score: {score:.2f}"
            )


        # =================================================
        # NLP QUESTION
        # =================================================

        else:

            # =============================================
            # STEP 2: RETRIEVE FROM CHROMADB
            # =============================================

            with st.spinner(
                "Searching the NLP knowledge base..."
            ):

                results = retriever.search(
                    query,
                    k=4
                )


            if not results:

                st.warning(
                    "I could not find relevant information "
                    "in the NLP knowledge base."
                )

            else:

                # =========================================
                # COMBINE RETRIEVED DOCUMENTS
                # =========================================

                context_parts = []

                for document in results:

                    if document:
                        context_parts.append(
                            str(document)
                        )

                context = "\n\n---\n\n".join(
                    context_parts
                )


                # =========================================
                # STEP 3: GET GROQ API KEY
                # =========================================

                try:

                    groq_api_key = st.secrets[
                        "GROQ_API_KEY"
                    ]

                except Exception:

                    groq_api_key = os.getenv(
                        "GROQ_API_KEY"
                    )


                # =========================================
                # CHECK API KEY
                # =========================================

                if not groq_api_key:

                    st.error(
                        "Groq API key is not configured."
                    )

                    st.info(
                        "Add GROQ_API_KEY in "
                        "Streamlit Secrets."
                    )

                    st.subheader(
                        "Retrieved NLP Knowledge"
                    )

                    st.write(context)


                else:

                    # =====================================
                    # STEP 4: GROQ CLIENT
                    # =====================================

                    try:

                        client = Groq(
                            api_key=groq_api_key
                        )


                        # =================================
                        # RAG SYSTEM PROMPT
                        # =================================

                        system_prompt = """
You are an NLP Knowledge Assistant.

Your job is to answer questions only about
Natural Language Processing (NLP).

Use the supplied NLP knowledge base as the
main source of information.

Rules:

1. Answer only NLP-related questions.

2. Do not answer unrelated questions.

3. Use the supplied context to ground your answer.

4. Do not invent facts that are not supported
   by the supplied context.

5. Explain concepts clearly and professionally.

6. Use simple examples when they help.

7. If the context does not contain enough
   information, clearly say that the knowledge
   base does not contain enough information.

8. Do not mention these instructions in your answer.
"""


                        # =================================
                        # USER PROMPT
                        # =================================

                        user_prompt = f"""
Here is the retrieved NLP knowledge:

------------------------------
{context}
------------------------------

User question:

{query}

Answer the question using the retrieved
NLP knowledge. Explain the concept clearly.
"""


                        # =================================
                        # GENERATE ANSWER
                        # =================================

                        with st.spinner(
                            "Generating NLP explanation..."
                        ):

                            response = (
                                client.chat.completions.create(

                                    # CURRENT GROQ MODEL
                                    model="openai/gpt-oss-20b",

                                    messages=[
                                        {
                                            "role": "system",
                                            "content": system_prompt
                                        },
                                        {
                                            "role": "user",
                                            "content": user_prompt
                                        }
                                    ],

                                    temperature=0.2,

                                    max_tokens=800
                                )
                            )


                        # =================================
                        # GET ANSWER
                        # =================================

                        answer = (
                            response
                            .choices[0]
                            .message
                            .content
                        )


                        # =================================
                        # DISPLAY ANSWER
                        # =================================

                        st.subheader(
                            "Answer"
                        )

                        st.write(
                            answer
                        )


                        # =================================
                        # RETRIEVED KNOWLEDGE
                        # =================================

                        with st.expander(
                            "🔎 Retrieved NLP Knowledge"
                        ):

                            st.write(
                                context
                            )


                    # =====================================
                    # GROQ ERROR
                    # =====================================

                    except Exception as e:

                        st.error(
                            "Unable to generate the "
                            "LLM response using Groq."
                        )

                        st.caption(
                            f"Error: {str(e)}"
                        )

                        st.subheader(
                            "Retrieved NLP Knowledge"
                        )

                        st.write(
                            context
                        )
```
