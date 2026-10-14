
import os
import uuid
import tempfile

import streamlit as st
import chromadb

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


# ---------------- CONFIG ----------------

load_dotenv()

st.set_page_config(
    page_title="PDF Analyzer",
    page_icon="📚",
    layout="wide"
)

API_KEY_GROQ = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-20b"


# ---------------- EMBEDDINGS ----------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class EmbeddingManager:
    def __init__(self):
        self.model = load_embedding_model()

    def generate_embeddings(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )


# ---------------- PDF LOADER ----------------

def load_uploaded_pdf(uploaded_file):
    documents = []

    # Save uploaded PDF temporarily.
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number + 1
                        }
                    )
                )
    finally:
        os.remove(temp_path)

    return documents


# ---------------- TEXT SPLITTER ----------------

def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)


# ---------------- VECTOR STORE ----------------

class VectorStoreManager:
    def __init__(self):
        # In-memory collection: no mixing with old PDFs.
        self.client = chromadb.EphemeralClient()

        self.collection = self.client.create_collection(
            name=f"pdf_{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError(
                "Documents and embeddings count do not match"
            )

        if not documents:
            return

        self.collection.add(
            ids=[
                f"chunk_{i}_{uuid.uuid4().hex}"
                for i in range(len(documents))
            ],
            documents=[
                doc.page_content for doc in documents
            ],
            metadatas=[
                doc.metadata for doc in documents
            ],
            embeddings=embeddings.tolist()
        )


# ---------------- RETRIEVER ----------------

class RAGRetriever:
    def __init__(self, embedding_manager, vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):
        query_embedding = (
            self.embedding_manager
            .generate_embeddings([query])[0]
        )

        total_docs = self.vector_store.collection.count()

        if total_docs == 0:
            return []

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total_docs)
        )

        retrieved_docs = []

        if results["documents"] and results["documents"][0]:
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                retrieved_docs.append({
                    "document": document,
                    "metadata": metadata,
                    "similarity_score": 1 - distance
                })

        return retrieved_docs


# ---------------- GENERATE ANSWER ----------------

def generate_output(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k)

    if not results:
        return (
            "Uploaded PDF mein relevant information nahi mili."
        ), []

    context = "\n\n".join(
        [
            (
                f"Source: {doc['metadata'].get('source', 'PDF')}, "
                f"Page: {doc['metadata'].get('page', '?')}\n"
                f"{doc['document']}"
            )
            for doc in results
        ]
    )

    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using only the provided context.
If the answer is not present in the context, say that the
uploaded PDF does not contain enough information.
Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    sources = [
        (
            f"{doc['metadata'].get('source', 'PDF')} "
            f"- Page {doc['metadata'].get('page', '?')}"
        )
        for doc in results
    ]

    return response.content, sources


# ---------------- STREAMLIT UI ----------------

st.title("📚 PDF Analyzer")
st.caption(
    "Upload a PDF and ask questions based on its content."
)

if not API_KEY_GROQ:
    st.error(
        "GROQ_API_KEY missing hai. Apni .env file check karo."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("👆 Pehle PDF upload karo, phir questions poochho.")
    st.stop()

# Process a newly uploaded PDF only once per session.
pdf_signature = (
    uploaded_file.name,
    len(uploaded_file.getvalue())
)

if st.session_state.get("pdf_signature") != pdf_signature:
    st.session_state.pop("rag_retriever", None)
    st.session_state.pop("chat_history", None)

    with st.spinner("Reading PDF and creating embeddings..."):
        try:
            documents = load_uploaded_pdf(uploaded_file)

            if not documents:
                st.error(
                    "PDF se text extract nahi hua. "
                    "Agar scanned PDF hai, OCR ki zaroorat ho sakti hai."
                )
                st.stop()

            chunks = split_docs(documents)

            embedding_manager = EmbeddingManager()
            embeddings = embedding_manager.generate_embeddings(
                [doc.page_content for doc in chunks]
            )

            vector_store = VectorStoreManager()
            vector_store.add_documents(chunks, embeddings)

            st.session_state.rag_retriever = RAGRetriever(
                embedding_manager,
                vector_store
            )

            st.session_state.pdf_signature = pdf_signature
            st.session_state.chat_history = []

        except Exception as error:
            st.error(f"PDF processing error: {error}")
            st.stop()

st.success(f"PDF ready: {uploaded_file.name}")

llm = ChatGroq(
    groq_api_key=API_KEY_GROQ,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=1024
)

# Show previous messages.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            st.caption(
                "Sources: " + ", ".join(message["sources"])
            )

query = st.chat_input("Ask a question about your PDF...")

if query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Finding answer in your PDF..."):
            try:
                answer, sources = generate_output(
                    query,
                    st.session_state.rag_retriever,
                    llm,
                    top_k=3
                )

                st.markdown(answer)

                if sources:
                    with st.expander("📄 View sources"):
                        for source in sources:
                            st.write(source)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:
                st.error(f"Answer generation failed: {error}")

