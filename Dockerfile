FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
# CPU-only torch keeps the image small
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -r requirements.txt
# Pre-download the embedding model so containers start offline
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY . .
RUN useradd -m appuser && mkdir -p /app/data /app/outputs && chown -R appuser /app
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.maxUploadSize=50"]