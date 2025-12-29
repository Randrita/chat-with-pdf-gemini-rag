import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from google import genai



# =========================
# ENV & GEMINI SETUP
# =========================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# =========================
# TEXT CHUNKING
# =========================
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# =========================
# VECTOR STORE (FAISS)
# =========================
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# =========================
# GEMINI ANSWER FUNCTION
# =========================
def ask_gemini(context, question):
    prompt = f"""
You are an assistant that answers questions ONLY using the given context.
If the answer is not present in the context, say:
"Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# =========================
# HANDLE USER QUESTION
# =========================
def handle_user_question(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    with st.spinner("🤖 Gemini is thinking..."):
        answer = ask_gemini(context, question)

    st.markdown(
        f"""
        <div class="chat-container">
            <div class="user-bubble">
                <strong>You:</strong><br>{question}
            </div>
            <div class="bot-bubble">
                <strong>Gemini:</strong><br>{answer}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("📄 View source text used for this answer"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(doc.page_content)
            st.markdown("---")




# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config(
        page_title="Chat with PDF using Gemini",
        layout="wide"
    )

    st.markdown(
        """
    <style>
    /* ---------- COMMON ---------- */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 10px;
    }

    .user-bubble, .bot-bubble {
        padding: 12px 16px;
        border-radius: 14px;
        max-width: 75%;
        font-size: 15px;
        line-height: 1.4;
        word-wrap: break-word;
    }

    /* ---------- LIGHT MODE ---------- */
    html[data-theme="light"] .user-bubble {
        align-self: flex-start;
        background-color: #f1f1f1;
        color: #000000;
    }

    html[data-theme="light"] .bot-bubble {
        align-self: flex-end;
        background-color: #e6f4ea;
        color: #1f7a1f;
        font-weight: 500;
    }

    /* ---------- DARK MODE ---------- */
    html[data-theme="dark"] .user-bubble {
        align-self: flex-start;
        background-color: #2b2b2b;
        color: #eaeaea;
        border: 1px solid #3a3a3a;
    }

    html[data-theme="dark"] .bot-bubble {
        align-self: flex-end;
        background-color: #0f3d2e;
        color: #9ef0c3;
        border: 1px solid #1f7a5a;
        font-weight: 500;
    }

    /* ---------- INPUT FIX (NO RED BORDER) ---------- */
    div[data-baseweb="input"] > div {
        border-color: #555 !important;
        background-color: transparent !important;
    }

    div[data-baseweb="input"] > div:hover {
        border-color: #4CAF50 !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #4CAF50 !important;
        box-shadow: none !important;
    }
    </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📄 Chat with PDF using Gemini 2.5 Flash")
    st.write("Upload PDF files and ask questions based only on their content.")

    user_question = st.text_input("Ask a question from the PDFs")

    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.header("📂 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload one or more PDF files",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = extract_pdf_text(pdf_docs)
                    chunks = split_text_into_chunks(raw_text)
                    create_vector_store(chunks)
                    st.success("✅ PDFs processed and indexed successfully!")



if __name__ == "__main__":
    main()
