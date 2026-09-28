import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

from nlp.scope_checker import ScopeChecker
from rag.retriever import Retriever


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NLP Knowledge Assistant",
    page_icon="🧠",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 20px;
}

.answer-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #f7f9fc;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "question" not in st.session_state:
    st.session_state.question = ""

if "recent_questions" not in st.session_state:
    st.session_state.recent_questions = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0


# ============================================================
# LOAD MODELS AND COMPONENTS
# ============================================================

@st.cache_resource
def load_components():

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # NLP scope checker
    scope_checker = ScopeChecker(
        model=model,
        threshold=0.38
    )

    # RAG retriever
    retriever = Retriever(
        embedding_model=model
    )

    return model, scope_checker, retriever


model, scope_checker, retriever = load_components()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 NLP Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions or learn NLP topics step-by-step'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 About the Project")

    st.write(
        "This assistant answers questions only from the "
        "Natural Language Processing domain."
    )

    st.markdown("---")

    st.subheader("🛠️ Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Sentence Transformers")
    st.write("• ChromaDB")
    st.write("• RAG")
    st.write("• Groq LLM")

    st.markdown("---")

    st.subheader("💡 Example Questions")

    example_questions = [
        "What is NLP?",
        "What is tokenization?",
        "How does TF-IDF work?",
        "What is sentiment analysis?",
        "What is BERT?",
        "What is self-attention?",
        "What is RAG?"
    ]

    for example in example_questions:

        if st.button(
            example,
            key=f"example_{example}"
        ):
            st.session_state.question = example
            st.session_state.input_key += 1
            st.rerun()

    st.markdown("---")

    if st.session_state.recent_questions:

        st.subheader("🕘 Recent Questions")

        for recent in reversed(
            st.session_state.recent_questions[-5:]
        ):
            st.caption(f"• {recent}")


# ============================================================
# LEARN THIS TOPIC
# ============================================================

st.markdown(
    '<div class="section-title">📖 Learn an NLP Topic</div>',
    unsafe_allow_html=True
)

topics = [
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
    topics
)

learn_button = st.button(
    "📖 Learn This Topic",
    type="secondary"
)

st.markdown("---")


# ============================================================
# QUESTION INPUT
# ============================================================

query = st.text_input(
    "Ask an NLP question:",
    value=st.session_state.question,
    placeholder="Example: What is tokenization?",
    key=f"question_input_{st.session_state.input_key}"
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary"
    )

with col2:

    clear_button = st.button(
        "🗑️ Clear"
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_button:

    st.session_state.question = ""
    st.session_state.input_key += 1

    st.rerun()


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    '<div class="section-title">💡 Suggested Questions</div>',
    unsafe_allow_html=True
)

suggestions = [
    "What is Natural Language Processing?",
    "What are the types of NLP?",
    "What is tokenization?",
    "What is TF-IDF?",
    "How does sentiment analysis work?",
    "What is BERT?",
    "What is RAG?"
]

suggestion_columns = st.columns(2)

for index, suggestion in enumerate(suggestions):

    with suggestion_columns[index % 2]:

        if st.button(
            suggestion,
            key=f"suggestion_{index}"
        ):

            st.session_state.question = suggestion
            st.session_state.input_key += 1

            st.rerun()


# ============================================================
# GROQ ANSWER FUNCTION
# ============================================================

def generate_nlp_answer(
    question,
    context,
    learn_mode=False,
    topic=None
):

    # --------------------------------------------------------
    # GROQ API KEY
    # --------------------------------------------------------

    try:
        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:

        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        return (
            "Groq API key was not found. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )


    # --------------------------------------------------------
    # CREATE GROQ CLIENT
    # --------------------------------------------------------

    client = Groq(
        api_key=api_key
    )


    # ========================================================
    # NORMAL QUESTION MODE
    # ========================================================

    if not learn_mode:

        system_prompt = """
You are an NLP Knowledge Assistant.

Your job is to answer questions about Natural Language Processing.

Use the supplied NLP knowledge base as the main source for your answer.

Important rules:

1. Answer only NLP-related questions.
2. Use the retrieved knowledge provided to you.
3. Do not invent information that is not supported by the knowledge base.
4. Explain concepts clearly and simply.
5. Give examples when useful.
6. If the retrieved knowledge does not contain enough information,
   clearly say that the available knowledge base does not contain
   enough information.
7. Do not answer unrelated topics.
"""

        user_prompt = f"""
Retrieved NLP Knowledge:

{context}


User Question:

{question}


Answer the question clearly using the retrieved NLP knowledge.
"""


    # ========================================================
    # LEARN MODE
    # ========================================================

    else:

        system_prompt = """
You are an NLP teacher inside an NLP Knowledge Assistant.

The user has selected a topic and wants to LEARN the topic.

This is different from simply answering one question.

Create a beginner-friendly mini lesson about the selected NLP topic.

Use the supplied NLP knowledge base as your main source.

Structure the lesson using the following sections whenever
the retrieved knowledge supports them:

1. What is it?
2. Why is it used?
3. How does it work?
4. Main components or concepts
5. Types or variations
6. Simple example
7. Applications
8. Advantages
9. Limitations
10. Quick recap

Important rules:

- Explain the topic in simple language.
- Teach the concept step-by-step.
- Do not make up types, applications, advantages, or limitations.
- Only include sections when the retrieved knowledge supports them.
- If a section is not supported by the knowledge base,
  leave it out rather than inventing information.
- Use examples from the knowledge base when available.
- Keep the explanation useful for a college student learning NLP.
- Do not turn the answer into a one-line definition.
- The goal is to TEACH the complete concept.
"""

        user_prompt = f"""
Topic to Learn:

{topic}


Retrieved NLP Knowledge:

{context}


Teach me this topic as a complete beginner-friendly NLP lesson.

Explain what it is, why it is used, how it works, its main concepts,
types or variations when available, examples, applications,
advantages and limitations when supported by the retrieved knowledge,
and finish with a short recap.

Use only the retrieved NLP knowledge as your factual source.
"""


    # ========================================================
    # GROQ REQUEST
    # ========================================================

    try:

        response = client.chat.completions.create(

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
            max_tokens=1200
        )

        return response.choices[0].message.content


    except Exception as e:

        return f"Groq error: {str(e)}"


# ============================================================
# PROCESS NORMAL QUESTION
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "Please enter an NLP question first."
        )

    else:

        st.session_state.question = query

        # ----------------------------------------------------
        # NLP SCOPE CHECK
        # ----------------------------------------------------

        in_scope, score = scope_checker.check(query)


        if not in_scope:

            st.warning(
                "⚠️ This question appears to be outside "
                "the NLP context."
            )

            st.caption(
                f"NLP relevance score: {score:.0%}"
            )


        else:

            st.success(
                "✅ Your question is related to NLP."
            )

            st.caption(
                f"NLP relevance score: {score:.0%}"
            )


            # ------------------------------------------------
            # RAG RETRIEVAL
            # ------------------------------------------------

            with st.spinner(
                "🔎 Searching the NLP knowledge base..."
            ):

                results = retriever.search(
                    query,
                    k=4
                )


            if results:

                context = "\n\n---\n\n".join(results)

            else:

                context = (
                    "No relevant information was retrieved "
                    "from the NLP knowledge base."
                )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "🤖 Generating your answer..."
            ):

                answer = generate_nlp_answer(
                    question=query,
                    context=context,
                    learn_mode=False
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown("## 🤖 AI Answer")

            st.markdown(
                '<div class="answer-box">',
                unsafe_allow_html=True
            )

            st.markdown(answer)

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # RETRIEVED KNOWLEDGE
            # ------------------------------------------------

            with st.expander(
                "📚 View Retrieved NLP Knowledge"
            ):

                st.write(
                    f"Found {len(results)} relevant sections."
                )

                for result in results:

                    st.markdown("---")
                    st.markdown(result)


            # ------------------------------------------------
            # RECENT QUESTIONS
            # ------------------------------------------------

            if query not in st.session_state.recent_questions:

                st.session_state.recent_questions.append(
                    query
                )


# ============================================================
# PROCESS LEARN TOPIC
# ============================================================

if learn_button:

    topic_question = f"Explain {selected_topic}"

    # --------------------------------------------------------
    # RETRIEVE KNOWLEDGE
    # --------------------------------------------------------

    with st.spinner(
        f"🔎 Collecting knowledge about {selected_topic}..."
    ):

        results = retriever.search(
            topic_question,
            k=4
        )


    if results:

        context = "\n\n---\n\n".join(results)

    else:

        context = (
            "No relevant information was retrieved "
            "from the NLP knowledge base."
        )


    # --------------------------------------------------------
    # GENERATE LESSON
    # --------------------------------------------------------

    with st.spinner(
        f"📖 Preparing your {selected_topic} lesson..."
    ):

        answer = generate_nlp_answer(
            question=topic_question,
            context=context,
            learn_mode=True,
            topic=selected_topic
        )


    # --------------------------------------------------------
    # DISPLAY LESSON
    # --------------------------------------------------------

    st.markdown(
        f"## 📖 Learn: {selected_topic}"
    )

    st.markdown(
        '<div class="answer-box">',
        unsafe_allow_html=True
    )

    st.markdown(answer)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # RETRIEVED KNOWLEDGE
    # --------------------------------------------------------

    with st.expander(
        "📚 View Retrieved NLP Knowledge"
    ):

        st.write(
            f"Found {len(results)} relevant sections."
        )

        for result in results:

            st.markdown("---")
            st.markdown(result)


    # --------------------------------------------------------
    # ADD TO RECENT QUESTIONS
    # --------------------------------------------------------

    learn_history = f"Learn: {selected_topic}"

    if learn_history not in st.session_state.recent_questions:

        st.session_state.recent_questions.append(
            learn_history
        )
