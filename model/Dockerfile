FROM python:3.11-slim

# git is required to pip-install CLIP from GitHub; gcc for any source builds
RUN apt-get update && apt-get install -y --no-install-recommends gcc git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_space.txt .
RUN pip install --no-cache-dir -r requirements_space.txt

# Copy ONLY the files the API needs — no notebooks, no checkpoint, no frontend
COPY rate_limit_engine.py .
COPY app.py .

RUN mkdir -p checkpoints

# HF Spaces requires a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Model loading: set MODEL_REPO_ID in Space variables so app.py pulls weights
# from HF Hub at startup. Local fallback path is used only when it is unset.
ENV MODEL_FILENAME="farl_classifier.pt" MODEL_PATH="checkpoints/farl_classifier.pt"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
