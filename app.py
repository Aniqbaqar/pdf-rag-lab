import streamlit as st
from rag_engine import PDFRAG, DEFAULT_CHAT_MODEL

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="centered")
st.title("📚 PDF Question Answering — RAG Lab")
st.write(
    "Upload a text-based PDF, build a searchable index, and ask questions. "
    "This demo uses local embeddings plus a free-tier Groq API model."
)

with st.expander("What is happening behind the scenes?"):
    st.markdown(
        "1. **Read PDF** — extract text from each page.\n"
        "2. **Chunk** — divide text into overlapping passages.\n"
        "3. **Embed** — convert passages into vectors.\n"
        "4. **Retrieve** — find passages closest to your question.\n"
        "5. **Generate** — ask the language model to answer from those passages."
    )

api_key = st.text_input(
    "Groq API key (not stored by this app)", type="password",
    help="Create a key in the Groq developer console. Free-tier limits apply."
)
model = st.text_input("Chat model ID", value=DEFAULT_CHAT_MODEL)
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if "rag" not in st.session_state:
    st.session_state.rag = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat" not in st.session_state:
    st.session_state.chat = []

if uploaded_pdf is not None:
    if st.button("Process PDF / Build RAG index", type="primary"):
        with st.spinner("Reading PDF and creating embeddings. First run may download the embedding model..."):
            try:
                rag = PDFRAG()
                count = rag.build(uploaded_pdf.getvalue())
                st.session_state.rag = rag
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.chat = []
                st.success(f"Ready! Indexed {count} text chunks from {uploaded_pdf.name}.")
            except Exception as exc:
                st.error(f"Could not process this PDF: {exc}")

if st.session_state.rag is not None:
    st.caption(f"Current document: {st.session_state.pdf_name}")
    question = st.text_input("Ask a question about your PDF", key="question")
    if st.button("Ask", type="primary"):
        if not api_key.strip():
            st.warning("Enter your Groq API key first.")
        elif not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving passages and generating an answer..."):
                try:
                    answer, passages = st.session_state.rag.answer(
                        question.strip(), api_key.strip(), model.strip()
                    )
                    st.session_state.chat.append(
                        {"question": question.strip(), "answer": answer, "passages": passages}
                    )
                except Exception as exc:
                    st.error(
                        "The request failed. Check the API key, model ID, internet "
                        f"connection, and provider limits. Details: {exc}"
                    )

    for item in reversed(st.session_state.chat):
        st.markdown("---")
        st.markdown(f"**Question:** {item['question']}")
        st.markdown(f"**Answer:** {item['answer']}")
        with st.expander("View retrieved PDF passages"):
            for i, passage in enumerate(item["passages"], start=1):
                st.markdown(f"**Passage {i}** · similarity {passage['score']:.3f}")
                st.write(passage["text"])

st.info(
    "Privacy note: the PDF is processed in the app session; retrieved passages and "
    "your question are sent to the selected LLM provider when you click Ask. "
    "Do not upload confidential student or personal documents."
)
