import os
import streamlit as st
import requests
import fitz
import docx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import *

# =============================
# Utils
# =============================
def extract_pdf_text(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
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
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index, embeddings, chunks

def load_index(index_path=INDEX_PATH):
    if not os.path.exists(index_path):
        return None, None
    index = faiss.read_index(index_path)
    return index

def search_chunks(query, chunks, index, top_k=TOP_K):
    model = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

# =============================
# Call Groq API
# =============================
def ask_groq(chunks, question):
    context = "\n\n".join(chunks)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý đọc hiểu tài liệu."},
            {"role": "user", "content": f"Ngữ cảnh: {context}\n\nCâu hỏi: {question}"}
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Lỗi API: {response.status_code} - {response.text}"

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="Trợ lý Tài liệu", page_icon="📚", layout="wide")
st.title("📚 Trợ lý Hỏi đáp Tài liệu (Grok API + FAISS)")

mode = st.radio("Chọn nguồn dữ liệu:", ["Google Drive (mặc định)", "Upload thủ công"])

if st.button("🔄 Đồng bộ lại OCR dữ liệu"):
    st.info("Đang đồng bộ lại dữ liệu...")

    os.makedirs("data", exist_ok=True)

    all_text = ""

    if mode == "Google Drive (mặc định)":
        # TODO: tải file từ Google Drive folder (cần API key / pydrive / gdown)
        # Ví dụ: bạn implement gdown.download_folder(GOOGLE_DRIVE_FOLDER)
        st.warning("🚧 Chưa implement lấy file từ Google Drive (cần API Google Drive hoặc gdown).")
    else:
        uploaded_files = st.file_uploader("Tải file PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_path.endswith(".pdf"):
                    all_text += extract_pdf_text(file_path)
                elif file_path.endswith(".docx"):
                    all_text += extract_docx_text(file_path)

    chunks = chunk_text(all_text)
    index, embeddings, saved_chunks = build_index(chunks)
    np.save("index/chunks.npy", saved_chunks)
    st.success("✅ Đồng bộ thành công!")

question = st.text_area("Nhập câu hỏi của bạn", height=100)
if st.button("🚀 Hỏi tài liệu"):
    if not os.path.exists(INDEX_PATH):
        st.error("⚠️ Chưa có dữ liệu, hãy đồng bộ trước.")
    else:
        index = load_index()
        chunks = np.load("index/chunks.npy", allow_pickle=True)
        relevant_chunks = search_chunks(question, chunks, index)
        answer = ask_groq(relevant_chunks, question)
        st.subheader("💡 Trả lời:")
        st.write(answer)
