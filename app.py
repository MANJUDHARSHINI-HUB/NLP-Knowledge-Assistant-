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

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

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
# LEARN A TOPIC FEATURE
# =========================================================

st.subheader("📖 Learn an NLP Topic")

topic_options = [
    "NLP",
    "Tokenization",
    "TF-IDF",
    "Word2Vec",
    "Sentiment Analysis",
    "BERT",
    "GPT",
    "Self-Attention",
    "Embeddings",
    "ChromaDB",
    "RAG",
    "Transformers"
]

selected_topic = st.selectbox(
    "Choose a topic to learn:",
    topic_options
)

learn_button = st.button(
    "📖 Learn This Topic",
    use_container_width=True
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
        st.session_state.question_input = "What is NLP?"

    if st.button(
        "What is Tokenization?",
        use_container_width=True
    ):
        st.session_state.question_input = "What is tokenization?"

    if st.button(
        "Explain BERT",
        use_container_width=True
    ):
        st.session_state.question_input = "Explain BERT"


with col2:

    if st.button(
        "What is RAG?",
        use_container_width=True
    ):
        st.session_state.question_input = "What is RAG?"

    if st.button(
        "Explain TF-IDF",
        use_container_width=True
    ):
        st.session_state.question_input = "Explain TF-IDF"

    if st.button(
        "What are embeddings?",
        use_container_width=True
    ):
        st.session_state.question_input = "What are embeddings?"


# =========================================================
# USER INPUT
# =========================================================

query = st.text_input(
    "💬 Ask your NLP question:",
    key="question_input",
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

    st.session_state.question_input = ""
    st.session_state.recent_questions = []

    st.rerun()


# =========================================================
# FUNCTION TO GENERATE ANSWER
# =========================================================

def generate_nlp_answer(user_question):

    # -----------------------------------------------------
    # STEP 1: CHECK NLP SCOPE
    # -----------------------------------------------------

    with st.spinner(
        "🔎 Checking whether your question is NLP-related..."
    ):

        in_scope, score = scope_checker.check(
            user_question
        )


    # -----------------------------------------------------
    # OUT OF SCOPE
    # -----------------------------------------------------

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

        return


    # -----------------------------------------------------
    # NLP QUESTION
    # -----------------------------------------------------

    st.success(
        "✅ Your question is related to NLP."
    )

    st.metric(
        "📊 NLP Relevance",
        f"{score * 100:.1f}%"
    )


    # -----------------------------------------------------
    # STEP 2: RETRIEVE KNOWLEDGE
    # -----------------------------------------------------

    with st.spinner(
        "📚 Searching the NLP knowledge base..."
    ):

        results = retriever.search(
            user_question,
            k=4
        )


    # -----------------------------------------------------
    # CHECK RETRIEVAL
    # -----------------------------------------------------

    if not results:

        st.warning(
            "I could not find relevant information "
            "in the NLP knowledge base."
        )

        return


    st.info(
        f"📚 {len(results)} relevant knowledge "
        f"section(s) found."
    )


    # -----------------------------------------------------
    # COMBINE RETRIEVED DOCUMENTS
    # -----------------------------------------------------

    context_parts = []

    for document in results:

        if document:

            context_parts.append(
                str(document)
            )

    context = "\n\n---\n\n".join(
        context_parts
    )


    # -----------------------------------------------------
    # GET GROQ API KEY
    # -----------------------------------------------------

    try:

        groq_api_key = st.secrets[
            "GROQ_API_KEY"
        ]

    except Exception:

        groq_api_key = os.getenv(
            "GROQ_API_KEY"
        )


    # -----------------------------------------------------
    # CHECK API KEY
    # -----------------------------------------------------

    if not groq_api_key:

        st.error(
            "Groq API key is not configured."
        )

        st.info(
            "Add GROQ_API_KEY in Streamlit Secrets."
        )

        return


    # -----------------------------------------------------
    # CREATE GROQ CLIENT
    # -----------------------------------------------------

    client = Groq(
        api_key=groq_api_key
    )


    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # USER PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
Retrieved NLP Knowledge:

{context}


User Question:

{user_question}


Please answer the user's question using
the retrieved NLP knowledge.

Explain the concept clearly and simply.
"""


    # -----------------------------------------------------
    # CALL GROQ LLM
    # -----------------------------------------------------

    try:

        with st.spinner(
            "🤖 Generating your NLP answer..."
        ):

            response = client.chat.completions.create(

                # SAME MODEL
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


        # -------------------------------------------------
        # GET ANSWER
        # -------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        st.divider()

        st.subheader(
            "🤖 AI Answer"
        )

        st.write(answer)


        # -------------------------------------------------
        # KNOWLEDGE USED
        # -------------------------------------------------

        with st.expander(
            "📚 View Knowledge Used"
        ):

            st.write(
                "The following information "
                "was retrieved from the NLP knowledge base:"
            )

            st.write(context)


    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    except Exception as e:

        st.error(
            "The AI service could not generate the answer."
        )

        st.info(
            "Here is the relevant NLP knowledge "
            "retrieved from the knowledge base:"
        )

        st.write(context)

        st.caption(
            f"Error: {e}"
        )


# =========================================================
# PROCESS NORMAL QUESTION
# =========================================================

if ask_button:

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # Save recent question
        if query not in st.session_state.recent_questions:

            st.session_state.recent_questions.insert(
                0,
                query
            )

        # Keep only latest 5
        st.session_state.recent_questions = (
            st.session_state.recent_questions[:5]
        )

        generate_nlp_answer(query)


# =========================================================
# PROCESS LEARN TOPIC
# =========================================================

if learn_button:

    topic_question = f"Explain {selected_topic}"

    st.session_state.question_input = topic_question

    # Save topic in recent questions
    if topic_question not in st.session_state.recent_questions:

        st.session_state.recent_questions.insert(
            0,
            topic_question
        )

    st.session_state.recent_questions = (
        st.session_state.recent_questions[:5]
    )

    st.subheader(
        f"📖 Learning: {selected_topic}"
    )

    generate_nlp_answer(topic_question)


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
