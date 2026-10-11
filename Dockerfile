FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Ollama runs on the host by default; override for Groq or a different host
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
