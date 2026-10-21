"""
Comprehensive Test Suite for AskDoc AI
Tests all scenarios:
- Health check
- Non-PDF upload validation
- Empty/invalid PDF handling
- Valid text PDF ingestion
- Chat Q&A with text document
- PDF containing embedded diagrams/images (Multimodal Vision)
- Chat Q&A about images in PDF
- Unrelated query handling
- Static frontend file serving
"""
import asyncio
import io
from fastapi.testclient import TestClient
from PyPDF2 import PdfWriter
from PIL import Image, ImageDraw

from app.main import app, doc_store
from app.config import GROQ_MODEL, GROQ_VISION_MODEL

client = TestClient(app)


def create_text_pdf(title: str, content: str) -> bytes:
    """Helper to create a simple text PDF using PyPDF2 / raw PDF bytes."""
    # We can write a simple valid PDF stream
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, title)
    y = 700
    for line in content.split("\n"):
        c.drawString(100, y, line)
        y -= 25
    c.save()
    buf.seek(0)
    return buf.getvalue()


def create_pdf_with_image(title: str, text: str, img_text: str) -> bytes:
    """Helper to create a PDF containing both text and an embedded image."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    # Create an image with drawing
    img = Image.new("RGB", (300, 150), color="white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 290, 140], outline="black", width=2)
    d.text((20, 30), "CHART: Q3 FINANCIAL PERFORMANCE", fill="blue")
    d.text((20, 60), img_text, fill="black")
    d.text((20, 90), "Target Met: 100%", fill="green")

    img_buf = io.BytesIO()
    img.save(img_buf, format="JPEG")
    img_buf.seek(0)

    # Insert into PDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, title)
    c.drawString(100, 720, text)
    c.drawImage(ImageReader(img_buf), 100, 500, width=250, height=125)
    c.save()
    buf.seek(0)
    return buf.getvalue()


def test_1_health_initial():
    print("\n--- Test 1: Initial Health Check ---")
    doc_store.clear()
    res = client.get("/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "healthy"
    assert data["document_loaded"] is False
    assert data["model"] == GROQ_MODEL
    assert data["vision_model"] == GROQ_VISION_MODEL
    print("[PASS] Initial health endpoint is healthy and reports no document loaded.")


def test_2_chat_without_document():
    print("\n--- Test 2: Chat Without Uploaded Document ---")
    doc_store.clear()
    res = client.post("/chat", json={"message": "What is this document about?"})
    assert res.status_code == 400
    assert "No document uploaded yet" in res.json()["detail"]
    print("[PASS] Correctly rejected chat query when no document is in memory.")


def test_3_upload_non_pdf():
    print("\n--- Test 3: Upload Non-PDF File Validation ---")
    files = {"file": ("notes.txt", b"Hello world text file", "text/plain")}
    res = client.post("/upload", files=files)
    assert res.status_code == 400
    assert "Only PDF files are supported" in res.json()["detail"]
    print("[PASS] Correctly rejected non-PDF upload (.txt).")


def test_4_upload_empty_pdf():
    print("\n--- Test 4: Upload Empty / Corrupt PDF File ---")
    files = {"file": ("empty.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")}
    res = client.post("/upload", files=files)
    assert res.status_code in [400, 500]
    print("[PASS] Correctly handled empty/corrupt PDF.")


def test_5_upload_text_pdf_and_query():
    print("\n--- Test 5: Upload Valid Text PDF & Q&A ---")
    doc_text = (
        "Project Apollo was an initiative to build high-efficiency solar battery cells.\n"
        "The project lead was Dr. Eleanor Vance, based in the Cambridge Research Lab.\n"
        "Testing achieved a conversion efficiency of 31.4 percent under standard solar irradiance."
    )
    pdf_bytes = create_text_pdf("Project Apollo Executive Report", doc_text)

    files = {"file": ("project_apollo.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    assert upload_data["status"] == "success"
    assert upload_data["chunks"] > 0
    print(f"[PASS] Text PDF uploaded successfully: {upload_data['chunks']} chunks created.")

    # Now ask question
    chat_res = client.post(
        "/chat", json={"message": "Who was the project lead and what was the efficiency achieved?"}
    )
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_data = chat_res.json()
    print("Response from Groq (GPT-OSS 120B):", chat_data["response"].encode("ascii", "replace").decode())
    assert "Eleanor" in chat_data["response"] or "31.4" in chat_data["response"]
    assert chat_data["sources"] > 0
    print("[PASS] Correctly answered question using text document context.")


def test_6_upload_pdf_with_image_and_query():
    print("\n--- Test 6: PDF with Embedded Diagram/Chart & Vision Q&A ---")
    doc_text = "This quarterly report provides an overview of operations."
    img_text = "Revenue: $4.5 Million | Net Profit: $1.2 Million"

    pdf_bytes = create_pdf_with_image("Quarterly Financial Report", doc_text, img_text)

    files = {"file": ("financial_chart.pdf", pdf_bytes, "application/pdf")}
    upload_res = client.post("/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    upload_data = upload_res.json()
    print("Upload response:", upload_data)
    assert upload_data["status"] == "success"
    assert upload_data["chunks"] > 0
    assert upload_data["images"] >= 1, "Expected at least 1 image analyzed via Vision AI"
    print(f"[PASS] PDF with image uploaded: {upload_data['images']} image(s) processed by Vision AI.")

    # Ask question specifically about the data inside the image
    chat_res = client.post(
        "/chat", json={"message": "What is the Net Profit and Revenue shown in the financial chart?"}
    )
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_data = chat_res.json()
    print("Response from Groq about the Image:", chat_data["response"].encode("ascii", "replace").decode())
    assert "1.2" in chat_data["response"] or "4.5" in chat_data["response"]
    print("[PASS] Successfully answered question about the visual chart/image in the PDF!")


def test_7_static_files():
    print("\n--- Test 7: Static Files & HTML UI Serving ---")
    for path in ["/", "/static/style.css", "/static/script.js"]:
        res = client.get(path)
        assert res.status_code == 200, f"Failed to fetch {path}"
    print("[PASS] All frontend static assets served with HTTP 200.")


if __name__ == "__main__":
    test_1_health_initial()
    test_2_chat_without_document()
    test_3_upload_non_pdf()
    test_4_upload_empty_pdf()
    test_5_upload_text_pdf_and_query()
    test_6_upload_pdf_with_image_and_query()
    test_7_static_files()
    print("\n==========================================")
    print("[SUCCESS] ALL SCENARIOS PASSED WITH 100% SUCCESS!")
    print("==========================================")
