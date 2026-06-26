---
title: Colour Season Analysis API
emoji: 🎨
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Colour Season Analysis — API

FastAPI inference backend for the Colour Season Analysis demo. Wraps a fine-tuned
CLIP ViT-B/16 classifier (4 colour seasons) plus FaRL face segmentation and
K-Means colour extraction. Stateless — the client sends an image, the API returns
the predicted season, confidence, per-region extracted colours, and a palette.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/` | Health check — `{"status": "ok", "model_loaded": true}` |
| `POST` | `/api/predict` | Body `{"image": "<base64 or data-URL>"}` → season + colours JSON |

## Configuration (Space → Settings → Variables and secrets)

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_REPO_ID` | yes | HF Hub model repo, e.g. `smubiainfrastructure/dap-ay2526-colour-analysis-model` |
| `MODEL_FILENAME` | no | Checkpoint filename in the repo (default `farl_classifier.pt`) |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins (default `*`; set to the Vercel URL) |
| `UPSTASH_REDIS_REST_URL` | no | Upstash REST URL — rate limiting disabled if unset |
| `UPSTASH_REDIS_REST_TOKEN` | no | Upstash REST token |
| `RATE_LIMIT_PREDICT_MAX` | no | Max `/api/predict` requests per window per IP (default `30`) |
| `RATE_LIMIT_WINDOW_SECONDS` | no | Window in seconds (default `60`) |
