```python
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

from nlp.scope_checker import ScopeChecker
from rag.retriever import Retriever


# ---------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="NLP Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 NLP Knowledge Assistant")

st.write(
    "Ask questions about Natural Language Processing (NLP), "
    "including concepts, techniques, algorithms, models, "
    "applications, Transformers, LLMs, embeddings, and RAG."
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# LOAD NLP COMPONENTS
# ---------------------------------------------------------

@st.cache_resource
def load_components():

    model = load_embedding_model()

    scope_checker = ScopeChecker(
        model=model,
        threshold=0.38
    )

    retriever = Retriever(
        model=model
    )

    return model, scope_checker, retriever


model, scope_checker, retriever = load_components()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📚 What can I ask?")

    st.write("You can ask questions such as:")

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
        "This assistant is specifically designed for "
        "NLP-related questions."
    )


# ---------------------------------------------------------
# USER QUESTION
# ---------------------------------------------------------

query = st.text_input(
    "💬 Ask your NLP question:",
    placeholder="Example: How does self-attention work?"
)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------

if st.button("Ask", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        # -------------------------------------------------
        # STEP 1: NLP SCOPE CHECK
        # -------------------------------------------------

        with st.spinner("Checking whether the question is NLP-related..."):

            in_scope, score = scope_checker.check(query)


        if not in_scope:

            st.warning(
                "Sorry, I’m an NLP-focused assistant. "
                "I can answer questions about NLP concepts, "
                "techniques, models, algorithms, and applications."
            )

            st.caption(
                f"NLP relevance score: {score:.2f}"
            )


        else:

            # -------------------------------------------------
            # STEP 2: RETRIEVE RELEVANT NLP KNOWLEDGE
            # -------------------------------------------------

            with st.spinner("Searching the NLP knowledge base..."):

                results = retriever.search(
                    query,
                    k=3
                )


            if not results:

                st.warning(
                    "I could not find relevant information "
                    "in the NLP knowledge base."
                )

            else:

                # -------------------------------------------------
                # COMBINE RETRIEVED DOCUMENTS
                # -------------------------------------------------

                context_parts = []

                for result in results:

                    if isinstance(result, dict):

                        document = result.get(
                            "document",
                            result.get("text", "")
                        )

                    else:

                        document = str(result)

                    if document:
                        context_parts.append(document)


                context = "\n\n---\n\n".join(
                    context_parts
                )


                # -------------------------------------------------
                # STEP 3: GROQ LLM
                # -------------------------------------------------

                try:

                    # Get API key from Streamlit Secrets
                    # or environment variable for local testing.

                    try:
                        groq_api_key = st.secrets["GROQ_API_KEY"]

                    except Exception:
                        groq_api_key = os.getenv(
                            "GROQ_API_KEY"
                        )


                    if not groq_api_key:

                        st.error(
                            "Groq API key is not configured. "
                            "Add GROQ_API_KEY in Streamlit Secrets."
                        )

                    else:

                        client = Groq(
                            api_key=groq_api_key
                        )


                        # -------------------------------------------------
                        # RAG PROMPT
                        # -------------------------------------------------

                        system_prompt = """
You are an NLP Knowledge Assistant.

Your job is to answer questions ONLY about
Natural Language Processing (NLP).

Use the supplied knowledge-base context as the
main source of information.

Rules:

1. Answer only NLP-related questions.
2. Do not answer unrelated questions.
3. Do not invent information that is not supported
   by the supplied context.
4. Explain concepts clearly and simply.
5. When useful, provide a small example.
6. If the supplied context does not contain enough
   information, clearly say that the knowledge base
   does not contain enough information.
7. Do not mention internal implementation details
   unless the user asks about the project itself.
"""


                        user_prompt = f"""
NLP KNOWLEDGE BASE:

{context}


USER QUESTION:

{query}


Using the knowledge above, answer the user's
question clearly and professionally.
"""


                        with st.spinner(
                            "Generating NLP explanation..."
                        ):

                            response = client.chat.completions.create(

                                model="llama-3.1-8b-instant",

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


                        answer = response.choices[0].message.content


                        # -------------------------------------------------
                        # STEP 4: DISPLAY ANSWER
                        # -------------------------------------------------

                        st.subheader("Answer")

                        st.write(answer)


                        # -------------------------------------------------
                        # RETRIEVED CONTEXT
                        # -------------------------------------------------

                        with st.expander(
                            "🔎 Retrieved NLP Knowledge"
                        ):

                            st.write(context)


                except Exception as e:

                    st.error(
                        "Unable to generate the answer using Groq."
                    )

                    st.caption(
                        f"Error: {str(e)}"
                    )

                    # Show retrieved information as fallback

                    st.subheader(
                        "Retrieved NLP Knowledge"
                    )

                    st.write(context)
```
