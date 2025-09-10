import os
import streamlit as st
import requests
import fitz
import docx
import faiss
import numpy as np
import gdown
from sentence_transformers import SentenceTransformer
from config import *

# =============================
# Utils
# =============================
def extract_pdf_text(file_path):
    """Đọc text từ PDF"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Lỗi đọc PDF: {e}")
    return text.strip()

def extract_docx_text(file_path):
    """Đọc text từ DOCX"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        st.error(f"Lỗi đọc DOCX: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chia nhỏ text thành các chunk"""
    chunks = []
    if not text.strip():
        return chunks
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# =============================
# FAISS Index
# =============================
def build_index(chunks, index_path=INDEX_PATH):
    """Tạo FAISS index từ list chunks"""
    if not chunks:
        return None, None, []
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    if embeddings.shape[0] == 0:
        return None, None, []
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    return index, embeddings, chunks

def load_index(index_path=INDEX_PATH):
    if not os.path.exists(index_path):
        return None
    return faiss.read_index(index_path)

def search_chunks(query, chunks, index, top_k=TOP_K):
    """Tìm top_k chunks liên quan nhất đến query"""
    if index is None or not chunks:
        return []
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# =============================
# Call Groq API
# =============================
def ask_groq(chunks, question):
    if not chunks:
        return "⚠️ Không có dữ liệu để hỏi đáp."
    context = "\n\n".join(chunks)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý đọc hiểu tài liệu."},
            {"role": "user", "content": f"Ngữ cảnh: {context}\n\nCâu hỏi: {question}"}
        ]
    }
    try:
        response = requests.post("{GROQ_API_URL}", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Lỗi API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Lỗi kết nối API: {e}"

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Trợ lý Tài liệu", page_icon="📚", layout="wide")
st.title("📚 Trợ lý Hỏi đáp Tài liệu (Grok API + FAISS)")

mode = st.radio("Chọn nguồn dữ liệu:", ["Google Drive (mặc định)", "Upload thủ công"])

if st.button("🔄 Đồng bộ lại OCR dữ liệu"):
    st.info("⏳ Đang đồng bộ lại dữ liệu...")

    os.makedirs("data", exist_ok=True)
    all_text = ""

    if mode == "Google Drive (mặc định)":
        try:
            st.write("📥 Đang tải file từ Google Drive...")
            gdown.download_folder(GOOGLE_DRIVE_FOLDER, output="data", quiet=False, use_cookies=False)
            for file_name in os.listdir("data"):
                file_path = os.path.join("data", file_name)
                if file_path.endswith(".pdf"):
                    all_text += extract_pdf_text(file_path) + "\n"
                elif file_path.endswith(".docx"):
                    all_text += extract_docx_text(file_path) + "\n"
        except Exception as e:
            st.error(f"❌ Lỗi tải file từ Google Drive: {e}")

    else:
        uploaded_files = st.file_uploader("Tải file PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_path.endswith(".pdf"):
                    all_text += extract_pdf_text(file_path) + "\n"
                elif file_path.endswith(".docx"):
                    all_text += extract_docx_text(file_path) + "\n"

    chunks = chunk_text(all_text)
    if not chunks:
        st.error("⚠️ Không tìm thấy text trong file. Có thể file toàn ảnh scan hoặc rỗng.")
    else:
        index, embeddings, saved_chunks = build_index(chunks)
        if index is None:
            st.error("⚠️ Không tạo được FAISS index.")
        else:
            np.save("index/chunks.npy", saved_chunks)
            st.success(f"✅ Đồng bộ thành công! ({len(saved_chunks)} chunks)")

# =============================
# Hỏi đáp
# =============================
question = st.text_area("Nhập câu hỏi của bạn", height=100)
if st.button("🚀 Hỏi tài liệu"):
    if not os.path.exists(INDEX_PATH) or not os.path.exists("index/chunks.npy"):
        st.error("⚠️ Chưa có dữ liệu, hãy đồng bộ trước.")
    else:
        index = load_index()
        chunks = np.load("index/chunks.npy", allow_pickle=True)
        relevant_chunks = search_chunks(question, chunks.tolist(), index)
        answer = ask_groq(relevant_chunks, question)
        st.subheader("💡 Trả lời:")
        st.write(answer)
