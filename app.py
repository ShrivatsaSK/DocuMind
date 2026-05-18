import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="DocuMind - Ask My Docs",
    page_icon="📄",
    layout="wide"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") #Your api key here 

st.title("📄 DocuMind - Ask My Docs")
st.caption("Hybrid Retrieval + Reranking + Citation-Enforced RAG")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Add it inside your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker


embedder, reranker = load_models()


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", "")
    return text.strip()


def extract_pdf(file):
    reader = PdfReader(file)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": page_num,
                "text": clean_text(text)
            })

    return pages


def extract_txt(file):
    text = file.read().decode("utf-8", errors="ignore")
    return [{"page": 1, "text": clean_text(text)}]


def sentence_chunk_text(text, max_words=160, overlap_words=35):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []

    for sentence in sentences:
        words = sentence.split()

        if len(current) + len(words) <= max_words:
            current.extend(words)
        else:
            if current:
                chunks.append(" ".join(current))

            current = current[-overlap_words:] + words

    if current:
        chunks.append(" ".join(current))

    return chunks

def process_documents(uploaded_files):
    chunks = []
    chunk_id = 1

    for file in uploaded_files:
        if file.name.lower().endswith(".pdf"):
            pages = extract_pdf(file)
        elif file.name.lower().endswith(".txt"):
            pages = extract_txt(file)
        else:
            continue

        for page in pages:
            text_chunks = sentence_chunk_text(page["text"])

            for chunk_text in text_chunks:
                chunks.append({
                    "id": chunk_id,
                    "source": file.name,
                    "page": page["page"],
                    "text": chunk_text
                })
                chunk_id += 1

    return chunks


def build_indexes(chunks):
    texts = [chunk["text"] for chunk in chunks]

    tokenized_docs = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized_docs)

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return bm25, embeddings


def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)

    if scores.max() - scores.min() == 0:
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def cosine_similarity_matrix(query_embedding, doc_embeddings):
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def hybrid_retrieve(query, chunks, bm25, embeddings, top_k=20):
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    query_embedding = embedder.encode(query, convert_to_numpy=True)
    vector_scores = cosine_similarity_matrix(query_embedding, embeddings)

    bm25_scores = normalize_scores(bm25_scores)
    vector_scores = normalize_scores(vector_scores)

    final_scores = (0.45 * bm25_scores) + (0.55 * vector_scores)

    top_indices = final_scores.argsort()[-top_k:][::-1]

    retrieved = []
    for idx in top_indices:
        item = chunks[idx].copy()
        item["retrieval_score"] = float(final_scores[idx])
        retrieved.append(item)

    return retrieved


def rerank_chunks(query, retrieved_chunks, top_k=5):
    pairs = [[query, chunk["text"]] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(retrieved_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_chunks = []

    for chunk, score in ranked[:top_k]:
        item = chunk.copy()
        item["rerank_score"] = float(score)
        final_chunks.append(item)

    return final_chunks


def build_context(chunks):
    context = ""

    for i, chunk in enumerate(chunks, start=1):
        context += f"""
[Source {i}]
Document: {chunk["source"]}
Page: {chunk["page"]}
Text: {chunk["text"]}
"""

    return context


def generate_answer(question, final_chunks):
    context = build_context(final_chunks)

    prompt = f"""
You are DocuMind, a strict document question-answering assistant.

Rules:
1. Answer only using the provided sources.
2. Use citations exactly like [Source 1], [Source 2].
3. Do not invent facts.
4. If the answer is not available in the sources, say:
   "I could not find this in the uploaded documents."
5. Keep the answer clear and useful.

Sources:
{context}

Question:
{question}

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


def citation_check(answer):
    return bool(re.search(r"\[Source\s+\d+\]", answer))


with st.sidebar:
    st.header("⚙️ Settings")

    top_k_retrieval = st.slider("Hybrid retrieval chunks", 5, 30, 20)
    top_k_rerank = st.slider("Final reranked chunks", 3, 10, 5)

    st.markdown("---")
    st.write("Recommended:")
    st.write("- Retrieval: 20")
    st.write("- Rerank: 5")


uploaded_files = st.file_uploader(
    "Upload PDF/TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Extracting text, creating chunks, and building indexes..."):
            chunks = process_documents(uploaded_files)

            if not chunks:
                st.error("No readable text found in uploaded documents.")
                st.stop()

            bm25, embeddings = build_indexes(chunks)

            st.session_state.chunks = chunks
            st.session_state.bm25 = bm25
            st.session_state.embeddings = embeddings

        st.success(f"Processed {len(chunks)} chunks successfully.")

if st.session_state.chunks:
    st.info(f"Ready. Total chunks indexed: {len(st.session_state.chunks)}")

    question = st.text_input("Ask a question from your documents")

    if question:
        with st.spinner("Retrieving, reranking, and generating answer..."):
            retrieved = hybrid_retrieve(
                question,
                st.session_state.chunks,
                st.session_state.bm25,
                st.session_state.embeddings,
                top_k=top_k_retrieval
            )

            final_chunks = rerank_chunks(
                question,
                retrieved,
                top_k=top_k_rerank
            )

            answer = generate_answer(question, final_chunks)

        st.subheader("✅ Answer")
        st.write(answer)

        if citation_check(answer):
            st.success("Citation check passed.")
        else:
            st.warning("Citation check failed. The answer may not have proper source citation.")

        st.subheader("📌 Sources Used")

        for i, chunk in enumerate(final_chunks, start=1):
            with st.expander(
                f"[Source {i}] {chunk['source']} - Page {chunk['page']}"
            ):
                st.write(chunk["text"])
                st.caption(
                    f"Retrieval score: {chunk.get('retrieval_score', 0):.3f} | "
                    f"Rerank score: {chunk.get('rerank_score', 0):.3f}"
                )
else:
    st.warning("Upload documents and click Process Documents.")
