import os
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

st.set_page_config(page_title="Ask My Docs RAG", layout="wide")

st.title("📄 Ask My Docs - Production RAG Demo")
st.write("Upload your documents and ask questions with citations.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("Please add your GROQ_API_KEY in the .env file.")

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

embedder, reranker = load_models()

def read_pdf(file):
    reader = PdfReader(file)
    text_pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_pages.append({
                "page": i + 1,
                "text": text
            })
    return text_pages

def read_txt(file):
    text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}]

def chunk_text(text, chunk_size=700, overlap=120):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

def process_files(uploaded_files):
    all_chunks = []

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            pages = read_pdf(file)
        elif file.name.endswith(".txt"):
            pages = read_txt(file)
        else:
            continue

        for page in pages:
            chunks = chunk_text(page["text"])

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file.name,
                    "page": page["page"]
                })

    return all_chunks

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_retrieve(query, chunks, top_k=20):
    texts = [c["text"] for c in chunks]

    # BM25 search
    tokenized_docs = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(query.lower().split())

    # Vector search
    doc_embeddings = embedder.encode(texts)
    query_embedding = embedder.encode(query)

    vector_scores = [
        cosine_similarity(query_embedding, emb)
        for emb in doc_embeddings
    ]

    # Normalize scores
    bm25_scores = np.array(bm25_scores)
    vector_scores = np.array(vector_scores)

    if bm25_scores.max() != 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    if vector_scores.max() != 0:
        vector_scores = vector_scores / vector_scores.max()

    final_scores = 0.5 * bm25_scores + 0.5 * vector_scores

    top_indices = final_scores.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]

def rerank(query, retrieved_chunks, top_k=5):
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [item[0] for item in ranked[:top_k]]

def generate_answer(query, final_chunks):
    context = ""

    for i, chunk in enumerate(final_chunks):
        context += f"""
Source {i+1}:
Document: {chunk['source']}
Page: {chunk['page']}
Text: {chunk['text']}
"""

    prompt = f"""
You are a strict document question-answering assistant.

Rules:
1. Answer only using the provided context.
2. Every factual claim must include citation in this format: [Document Name, Page X].
3. If the answer is not present in the context, say: "I could not find this in the uploaded documents."
4. Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        chunks = process_files(uploaded_files)

    st.success(f"Processed {len(chunks)} text chunks.")

    question = st.text_input("Ask a question from your documents:")

    if question:
        with st.spinner("Searching and generating answer..."):
            retrieved = hybrid_retrieve(question, chunks)
            reranked = rerank(question, retrieved)
            answer = generate_answer(question, reranked)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources Used")
        for chunk in reranked:
            with st.expander(f"{chunk['source']} - Page {chunk['page']}"):
                st.write(chunk["text"])