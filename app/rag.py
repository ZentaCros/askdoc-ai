"""
AskDoc AI - RAG Pipeline
Handles PDF extraction, image analysis via Vision LLM, text chunking, and retrieval.
Uses TF-IDF for embeddings (no external vector DB needed) and cosine similarity for retrieval.
"""
import re
import math
from typing import List, Tuple


class DocumentStore:
    """In-memory document store with TF-IDF based retrieval and Multimodal Vision support."""

    def __init__(self):
        self.chunks: List[str] = []
        self.filename: str = ""
        self.tfidf_matrix: List[dict] = []
        self.idf: dict = {}
        self.vocab: set = set()
        self.images_analyzed: int = 0

    def clear(self):
        self.chunks = []
        self.filename = ""
        self.tfidf_matrix = []
        self.idf = {}
        self.vocab = set()
        self.images_analyzed = 0

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract raw text from PDF bytes using PyPDF2."""
        from PyPDF2 import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def _extract_images_from_page(self, page) -> List[Tuple[str, bytes]]:
        """Extract embedded images from a PDF page, handling all filter and wrapper variations."""
        images = []
        # 1. Try standard PyPDF2 images property
        try:
            if hasattr(page, "images") and page.images:
                for img in page.images:
                    if img.data and len(img.data) >= 2048:
                        images.append((img.name, img.data))
        except Exception:
            pass

        # 2. Inspect XObject resources directly to catch images wrapped in multi-filter streams
        try:
            if "/Resources" in page and "/XObject" in page["/Resources"]:
                x_object = page["/Resources"]["/XObject"].get_object()
                for name, obj_ref in x_object.items():
                    obj = obj_ref.get_object()
                    if obj.get("/Subtype") == "/Image":
                        data = obj.get_data()
                        if data and len(data) >= 2048:
                            clean_name = name.lstrip("/")
                            if not any(img_name == clean_name for img_name, _ in images):
                                images.append((clean_name, data))
        except Exception:
            pass

        return images

    async def _describe_image_async(
        self, img_bytes: bytes, client, api_key: str, vision_model: str
    ) -> str:
        """Analyze images, charts, and diagrams using Groq's multimodal vision model."""
        if not api_key:
            return ""
        try:
            import io
            import base64
            from PIL import Image

            img = Image.open(io.BytesIO(img_bytes))

            # Skip tiny images (e.g. icons, bullets, dividers)
            if img.width < 50 or img.height < 50:
                return ""

            # Resize if large to preserve bandwidth and latency
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024))

            if img.mode != "RGB":
                img = img.convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64}"

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Analyze this document figure/image/diagram. "
                                        "Extract all legible text, labels, numbers, tables, "
                                        "and summarize what it depicts concisely."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": data_uri},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2,
                },
                timeout=25.0,
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # Vision analysis failure must never crash document ingestion
            pass
        return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks respecting sentence boundaries."""
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

    async def ingest_pdf(
        self,
        pdf_bytes: bytes,
        filename: str,
        api_key: str = "",
        vision_model: str = "qwen/qwen3.8-27b",
        max_images: int = 5,
    ) -> Tuple[int, int]:
        """Process a PDF: extract text, transcribe images with Vision AI, chunk, and index."""
        self.clear()
        self.filename = filename

        from PyPDF2 import PdfReader
        from io import BytesIO
        import httpx

        reader = PdfReader(BytesIO(pdf_bytes))
        text_parts = []
        images_to_process = []

        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")

            # Extract embedded images from the page
            if api_key:
                page_imgs = self._extract_images_from_page(page)
                for img_name, img_data in page_imgs:
                    if len(images_to_process) >= max_images:
                        break
                    images_to_process.append((page_idx + 1, img_name, img_data))

        # Transcribe images via Multimodal Vision LLM
        images_described = 0
        if images_to_process and api_key:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for page_num, img_name, img_data in images_to_process:
                    desc = await self._describe_image_async(
                        img_data, client, api_key, vision_model
                    )
                    if desc:
                        images_described += 1
                        text_parts.append(
                            f"\n[Image/Figure on Page {page_num} ({img_name})]:\n{desc}\n"
                        )

        self.images_analyzed = images_described
        full_text = "\n\n".join(text_parts).strip()
        if not full_text:
            raise ValueError(
                "Could not extract text or images from PDF. The file may be empty or unreadable."
            )

        self.chunks = self.chunk_text(full_text)
        self._build_index()

        return len(self.chunks), self.images_analyzed

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
- Quote relevant text or reference figures/images when helpful"""
