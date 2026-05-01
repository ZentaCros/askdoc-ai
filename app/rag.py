"""
AskDoc AI - RAG Pipeline
Handles PDF extraction, text chunking, embedding, and retrieval.
Uses TF-IDF for embeddings (no external API needed) and cosine similarity for retrieval.
"""
import re
import math
from typing import List, Tuple


class DocumentStore:
    """In-memory document store with TF-IDF based retrieval."""

    def __init__(self):
        self.chunks: List[str] = []
        self.filename: str = ""
        self.tfidf_matrix: List[dict] = []
        self.idf: dict = {}
        self.vocab: set = set()

    def clear(self):
        self.chunks = []
        self.filename = ""
        self.tfidf_matrix = []
        self.idf = {}
        self.vocab = set()

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using PyPDF2."""
        from PyPDF2 import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    end = break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _compute_tf(self, tokens: List[str]) -> dict:
        """Compute term frequency for a list of tokens."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}

    def _build_index(self):
        """Build TF-IDF index from chunks."""
        n_docs = len(self.chunks)
        if n_docs == 0:
            return

        # Compute document frequencies
        df = {}
        all_tokens = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk)
            all_tokens.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
                self.vocab.add(token)

        # Compute IDF
        self.idf = {
            token: math.log(n_docs / (1 + freq))
            for token, freq in df.items()
        }

        # Compute TF-IDF for each chunk
        self.tfidf_matrix = []
        for tokens in all_tokens:
            tf = self._compute_tf(tokens)
            tfidf = {token: tf_val * self.idf.get(token, 0) for token, tf_val in tf.items()}
            self.tfidf_matrix.append(tfidf)

    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Process a PDF: extract text, chunk it, build index."""
        self.clear()
        self.filename = filename

        text = self.extract_text_from_pdf(pdf_bytes)
        if not text.strip():
            raise ValueError("Could not extract text from PDF. The file may be scanned/image-based.")

        self.chunks = self.chunk_text(text)
        self._build_index()

        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the most relevant chunks for a query."""
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = {token: tf_val * self.idf.get(token, 0) for token, tf_val in query_tf.items()}

        scores = []
        for i, chunk_tfidf in enumerate(self.tfidf_matrix):
            sim = self._cosine_similarity(query_tfidf, chunk_tfidf)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], score))

        return results


def build_rag_prompt(query: str, context_chunks: List[Tuple[str, float]]) -> str:
    """Build a prompt that includes retrieved context for the LLM."""
    if not context_chunks:
        return f"""The user asked a question but no relevant context was found in the uploaded document.

User Question: {query}

Please respond by saying you couldn't find relevant information in the document for this question, and suggest the user ask something related to the document's content."""

    context = "\n\n---\n\n".join([chunk for chunk, _ in context_chunks])

    return f"""You are AskDoc AI, a helpful document assistant. Answer the user's question based ONLY on the provided document context. If the answer is not found in the context, say so honestly.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be concise and accurate
- If the context doesn't contain the answer, say "I couldn't find this information in the uploaded document"
- Use bullet points for lists
- Quote relevant text when helpful"""
