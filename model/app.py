import os
import io
import base64

import torch
import torch.nn as nn
import clip
import facer
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rate_limit_engine import RateLimiter

# ── Config ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['autunno', 'estate', 'inverno', 'primavera']

# Model loading: weights live on HF Hub. Set MODEL_REPO_ID in Space variables.
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "farl_classifier.pt")
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/farl_classifier.pt")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

SEASON_INFO = {
    'autunno': {
        'english': 'Autumn',
        'description': 'Warm, rich, earthy tones. Golden yellows, burnt oranges, deep browns.',
        'colours': ['#8B4513', '#D2691E', '#CD853F', '#A0522D', '#6B3A2A']
    },
    'estate': {
        'english': 'Summer',
        'description': 'Cool, muted, soft tones. Dusty roses, lavender, soft blues.',
        'colours': ['#B0A0C0', '#C8A0A0', '#A0B4C8', '#D4B0C0', '#8899AA']
    },
    'inverno': {
        'english': 'Winter',
        'description': 'Cool, high contrast, vivid tones. Pure white, jet black, icy blues.',
        'colours': ['#1C1C2E', '#E8E8F0', '#4169E1', '#DC143C', '#2F4F4F']
    },
    'primavera': {
        'english': 'Spring',
        'description': 'Warm, bright, clear tones. Coral, peach, golden yellow, fresh greens.',
        'colours': ['#FF7F50', '#FFD700', '#90EE90', '#FFA07A', '#FFDAB9']
    }
}


# ── Model definition (must match training exactly) ──
class FaRLImageClassifier(nn.Module):
    def __init__(self, clip_base_model, num_classes=4):
        super().__init__()
        self.clip_base_model = clip_base_model
        for param in self.clip_base_model.parameters():
            param.requires_grad = False
        classifier_in_features = clip_base_model.text_projection.shape[1]
        self.classification_head = nn.Linear(classifier_in_features, num_classes)

    def forward(self, x):
        features = self.clip_base_model.encode_image(x.type(self.clip_base_model.dtype))
        return self.classification_head(features)


# ── Resolve the trained checkpoint (HF Hub, else local fallback) ──
def resolve_checkpoint_path():
    if MODEL_REPO_ID:
        from huggingface_hub import hf_hub_download
        print(f"Downloading checkpoint {MODEL_FILENAME} from {MODEL_REPO_ID}...")
        return hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    if os.path.exists(MODEL_PATH):
        print(f"MODEL_REPO_ID not set — using local checkpoint {MODEL_PATH}")
        return MODEL_PATH
    print("⚠️  No MODEL_REPO_ID and no local checkpoint — running with an UNTRAINED head.")
    return None


# ── Load models at startup ──
print("Loading CLIP...")
clip_model, preprocess = clip.load("ViT-B/16", device=device)

print("Building classifier...")
model = FaRLImageClassifier(clip_model, num_classes=len(CLASS_NAMES)).to(device)

MODEL_LOADED = False
_ckpt_path = resolve_checkpoint_path()
if _ckpt_path:
    print("Loading classifier weights...")
    checkpoint = torch.load(_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    MODEL_LOADED = True
model.eval()

print("Loading FaRL face parser...")
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)
print("✅ All models loaded!")


# ── Rate limiting (anti-spam — shared engine) ──
limiter = RateLimiter(
    project="dap-ay2526-colour-analysis",
    buckets={"predict": int(os.environ.get("RATE_LIMIT_PREDICT_MAX", "30"))},
    redis_url=os.environ.get("UPSTASH_REDIS_REST_URL", os.environ.get("UPSTASH_REDIS_URL", "")),
    redis_token=os.environ.get("UPSTASH_REDIS_REST_TOKEN", os.environ.get("UPSTASH_REDIS_TOKEN", "")),
    default_window=int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60")),
)


def check_rate_limit(request, bucket):
    limiter.enforce(request, bucket)


# ── Colour helpers ──
def to_hsv(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]


def to_lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]


def extract_region_color(img_np, mask):
    pixels = img_np[mask > 0].reshape(-1, 3)
    if len(pixels) < 20:
        return None
    n = min(5, max(2, len(pixels) // 20))
    km = KMeans(n_clusters=n, n_init=10, random_state=42)
    km.fit(pixels.astype(np.float32))
    counts = np.bincount(km.labels_)
    chosen = np.argmax(counts)
    return km.cluster_centers_[chosen].astype(np.uint8)


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ── Core inference (stateless) ──
def run_prediction(image: Image.Image):
    # Classify
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    season = CLASS_NAMES[pred_idx]
    info = SEASON_INFO[season]
    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASS_NAMES))}

    # Extract face colours
    colours = {"skin": None, "eyes": None, "lips": None}
    try:
        img_np = np.array(image.convert("RGB"))
        img_t = torch.tensor(img_np).to(device)
        img_b = facer.hwc2bchw(img_t)
        with torch.inference_mode():
            faces = face_detector(img_b)
            if faces and len(faces.get('rects', [])) > 0:
                faces = face_parser(img_b, faces)
                seg = faces['seg']['logits'].argmax(dim=1).squeeze().cpu().numpy()
                kernel = np.ones((3, 3), np.uint8)
                non_black = np.any(img_np > 0, axis=-1)

                skin_mask = cv2.erode(((seg == 1) & non_black).astype(np.uint8), kernel, iterations=2)
                eye_mask = cv2.erode((np.isin(seg, [4, 5]) & non_black).astype(np.uint8), kernel, iterations=1)
                lip_mask = cv2.erode((np.isin(seg, [7, 8, 9]) & non_black).astype(np.uint8), kernel, iterations=1)

                s = extract_region_color(img_np, skin_mask)
                e = extract_region_color(img_np, eye_mask)
                l = extract_region_color(img_np, lip_mask)

                if s is not None: colours['skin'] = rgb_to_hex(s)
                if e is not None: colours['eyes'] = rgb_to_hex(e)
                if l is not None: colours['lips'] = rgb_to_hex(l)
    except Exception as ex:
        print(f"Colour extraction failed: {ex}")

    return {
        "season": season,
        "english": info['english'],
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "probabilities": prob_dict,
        "description": info['description'],
        "palette": info['colours'],
        "colours": colours,
    }


def decode_image(data: str) -> Image.Image:
    """Accept a raw base64 string or a data-URL (data:image/...;base64,...)."""
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ── FastAPI app ──
app = FastAPI(title="Colour Season API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image: str  # base64 string or data-URL


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/api/predict")
def predict(req: PredictRequest, request: Request):
    check_rate_limit(request, "predict")
    image = decode_image(req.image)
    return run_prediction(image)
