# =============================
# Cấu hình hệ thống
# =============================

# Groq API
GROQ_API_URL = "https://api.groq.ai/v1/chat/completions"
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Google Drive
GOOGLE_DRIVE_FOLDER_ID = "1qYLaWRbRnC0j4cRX9jzeJgKqmn185tNm"  # thay bằng folder ID thật

# Embedding Model
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# FAISS
INDEX_PATH = "index/faiss.index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 3
