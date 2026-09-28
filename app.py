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
# TITLE
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

    model = load_embedding_model()

    scope_checker = ScopeChecker(
        model=model,
        threshold=0.38
    )

    retriever = Retriever(
        embedding_model=model
    )

    return model, scope_checker, retriever


model, scope_checker, retriever = load_components()


# =========================================================
# SESSION STATE
# =========================================================

if "question" not in st.session_state:
    st.session_state.question = ""

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []


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

    st.subheader("📊 Project Info")

    st.write("🧠 Domain: Natural Language Processing")
    st.write("🔍 Search: Semantic Search")
    st.write("📚 Database: ChromaDB")
    st.write("🔗 Method: RAG")
    st.write("🤖 AI: Groq LLM")
    st.write("🧩 Embedding: MiniLM")

    st.divider()

    st.info(
        "Questions unrelated to NLP are outside "
        "the scope of this assistant."
    )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

st.subheader("💡 Try a question")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "What is NLP?",
        use_container_width=True
    ):
        st.session_state.question = "What is NLP?"

    if st.button(
        "What is Tokenization?",
        use_container_width=True
    ):
        st.session_state.question = "What is tokenization?"

    if st.button(
        "Explain BERT",
        use_container_width=True
    ):
        st.session_state.question = "Explain BERT"


with col2:

    if st.button(
        "What is RAG?",
        use_container_width=True
    ):
        st.session_state.question = "What is RAG?"

    if st.button(
        "Explain TF-IDF",
        use_container_width=True
    ):
        st.session_state.question = "Explain TF-IDF"

    if st.button(
        "What are embeddings?",
        use_container_width=True
    ):
        st.session_state.question = "What are embeddings?"


# =========================================================
# USER INPUT
# =========================================================

query = st.text_input(
    "💬 Ask your NLP question:",
    value=st.session_state.question,
    placeholder="Example: What is tokenization?"
)


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    ask_button = st.button(
        "🔍 Ask",
        type="primary",
        use_container_width=True
    )

with col2:

    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True
    )


# =========================================================
# CLEAR BUTTON
# =========================================================

if clear_button:

    st.session_state.question = ""

    st.rerun()


# =========================================================
# ASK BUTTON
# =========================================================

if ask_button:

    # -----------------------------------------------------
    # CHECK EMPTY QUESTION
    # -----------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # =================================================
        # SAVE QUESTION
        # =================================================

        st.session_state.question = query

        if query not in st.session_state.recent_questions:

            st.session_state.recent_questions.insert(
                0,
                query
            )

        # Keep only the latest 5 questions

        st.session_state.recent_questions = (
            st.session_state.recent_questions[:5]
        )


        # =================================================
        # STEP 1: CHECK NLP SCOPE
        # =================================================

        with st.spinner(
            "🔎 Checking whether your question is NLP-related..."
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

            st.metric(
                "📊 NLP Relevance",
                f"{score * 100:.1f}%"
            )


        # =================================================
        # NLP QUESTION
        # =================================================

        else:

            # =============================================
            # NLP RELEVANCE
            # =============================================

            st.success(
                "✅ Your question is related to NLP."
            )

            st.metric(
                "📊 NLP Relevance",
                f"{score * 100:.1f}%"
            )


            # =============================================
            # STEP 2: RETRIEVE KNOWLEDGE
            # =============================================

            with st.spinner(
                "📚 Searching the NLP knowledge base..."
            ):

                results = retriever.search(
                    query,
                    k=4
                )


            # =============================================
            # CHECK RETRIEVAL
            # =============================================

            if not results:

                st.warning(
                    "I could not find relevant information "
                    "in the NLP knowledge base."
                )


            else:

                # =========================================
                # SHOW RETRIEVAL INFORMATION
                # =========================================

                st.info(
                    f"📚 {len(results)} relevant knowledge "
                    f"section(s) found."
                )


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


                else:

                    # =====================================
                    # CREATE GROQ CLIENT
                    # =====================================

                    client = Groq(
                        api_key=groq_api_key
                    )


                    # =====================================
                    # SYSTEM PROMPT
                    # =====================================

                    system_prompt = """
You are an NLP Knowledge Assistant.

Your job is to answer questions only about
Natural Language Processing.

Use the supplied knowledge base to support
your answer.

Give a clear and easy-to-understand explanation.

If the retrieved knowledge does not contain
enough information, say that clearly instead
of inventing unsupported facts.

Stay within the NLP domain.
"""


                    # =====================================
                    # USER PROMPT
                    # =====================================

                    user_prompt = f"""
Retrieved NLP Knowledge:

{context}


User Question:

{query}


Please answer the user's question using
the retrieved NLP knowledge.
"""


                    # =====================================
                    # CALL GROQ LLM
                    # =====================================

                    try:

                        with st.spinner(
                            "🤖 Generating your NLP answer..."
                        ):

                            response = client.chat.completions.create(

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

                        st.divider()

                        st.subheader(
                            "🤖 AI Answer"
                        )

                        st.write(answer)


                        # =================================
                        # KNOWLEDGE USED
                        # =================================

                        with st.expander(
                            "📚 View Knowledge Used"
                        ):

                            st.write(
                                "The following information "
                                "was retrieved from the "
                                "NLP knowledge base:"
                            )

                            st.write(context)


                    # =====================================
                    # ERROR HANDLING
                    # =====================================

                    except Exception as e:

                        st.error(
                            "The AI service could not "
                            "generate the answer."
                        )

                        st.info(
                            "Here is the relevant "
                            "NLP knowledge retrieved "
                            "from the knowledge base:"
                        )

                        st.write(context)

                        st.caption(
                            f"Error: {e}"
                        )


# =========================================================
# RECENT QUESTIONS
# =========================================================

if st.session_state.recent_questions:

    st.divider()

    st.subheader(
        "🕒 Recent Questions"
    )

    for i, question in enumerate(
        st.session_state.recent_questions,
        start=1
    ):

        st.write(
            f"{i}. {question}"
        )
