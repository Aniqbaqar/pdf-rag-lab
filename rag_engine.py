"""Small PDF RAG engine for a classroom lab.

RAG = Retrieval-Augmented Generation:
1. Extract text from an uploaded PDF.
2. Split text into chunks.
3. Embed chunks and retrieve the most relevant chunks for a question.
4. Send retrieved context to a free-tier LLM API (Groq).

Note: API providers have usage/rate limits. No hosted free model promises unlimited tokens.
"""
import io
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"

def read_pdf(pdf_bytes: bytes) -> str:
    """Extract selectable text from PDF bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. If this is a scanned PDF, run OCR first."
        )
    return text

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150):
    """Split text into overlapping character chunks for a simple lab demo."""
    text = " ".join(text.split())
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class PDFRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self.index = None

    def build(self, pdf_bytes: bytes):
        text = read_pdf(pdf_bytes)
        self.chunks = split_into_chunks(text)
        if not self.chunks:
            raise ValueError("No text chunks could be created from this PDF.")
        vectors = self.embedder.encode(
            self.chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return len(self.chunks)

    def retrieve(self, question: str, top_k: int = 4):
        if self.index is None:
            raise ValueError("Upload and process a PDF first.")
        query_vector = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(query_vector, k)
        return [
            {"text": self.chunks[int(i)], "score": float(score)}
            for score, i in zip(scores[0], ids[0]) if int(i) >= 0
        ]

    def answer(self, question: str, api_key: str, model: str = DEFAULT_CHAT_MODEL):
        matches = self.retrieve(question)
        context = "\n\n".join(
            f"[Passage {i+1}]\n{item['text']}" for i, item in enumerate(matches)
        )
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful teaching assistant. Answer using only the "
                        "provided PDF context. If the answer is not in the context, "
                        "say that the PDF does not provide enough information. "
                        "Mention the passage numbers that support your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"PDF context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content, matches
