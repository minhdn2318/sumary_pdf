import streamlit as st
import requests
import fitz  # PyMuPDF để đọc PDF

# ==========================
# Cấu hình API
# ==========================
GROK_API_URL = "https://api.grog.ai/v1/chat/completions"
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================
# Hàm đọc toàn bộ text trong PDF
# ==========================
def extract_pdf_text(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# ==========================
# Hàm gọi API Grok
# ==========================
def ask_grok(pdf_text, question):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý đọc hiểu PDF."},
            {"role": "user", "content": f"Tài liệu: {pdf_text}\n\nCâu hỏi: {question}"}
        ]
    }

    response = requests.post(GROK_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Lỗi API: {response.status_code} - {response.text}"

# ==========================
# Giao diện Streamlit
# ==========================
st.set_page_config(page_title="Trợ lý PDF", page_icon="📄", layout="centered")

st.title("📄 Trợ lý Hỏi đáp PDF (Grok API)")

uploaded_pdf = st.file_uploader("Tải file PDF lên", type="pdf")
question = st.text_area("Nhập câu hỏi của bạn", height=100)

if st.button("🚀 Hỏi PDF"):
    if uploaded_pdf is None:
        st.warning("Vui lòng tải lên file PDF trước.")
    elif not question.strip():
        st.warning("Vui lòng nhập câu hỏi.")
    else:
        with st.spinner("⏳ Đang đọc PDF..."):
            pdf_text = extract_pdf_text(uploaded_pdf)

        with st.spinner("🤖 Đang hỏi Grok..."):
            answer = ask_grok(pdf_text, question)

        st.subheader("💡 Trả lời:")
        st.write(answer)
