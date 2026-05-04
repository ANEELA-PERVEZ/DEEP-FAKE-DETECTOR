# -*- coding: utf-8 -*-
"""
Advanced Video Forgery Detector (Optimized for HF Spaces)
"""
import gradio as gr
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import gc
import requests
import base64
from io import BytesIO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from retinaface import RetinaFace
import open_clip

# --- 🚨 MASTER FIX FOR PYTORCH 2.6 SECURITY ---
import torch.serialization
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except:
    pass

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ----------------------------------------------

# --- 1. SETUP & AUTH ---
print("🚀 Initializing Advanced Voting System with OpenCLIP & Claude...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# API Key for Claude via OpenRouter securely fetched from HF Secrets
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_memory()

# --- 2. OPEN CLIP PROMPTS (15 Real, 15 Fake) ---
REAL_PROMPTS = [
    "authentic high quality photo", "natural skin texture", "real human face", 
    "genuine unaltered photograph", "natural facial imperfections", "realistic human portrait", 
    "unedited camera footage", "authentic facial expression", "natural lighting and shadows", 
    "real person talking", "unmanipulated video frame", "genuine human skin", 
    "natural hair physics", "realistic eye reflections", "authentic human features"
]

FAKE_PROMPTS = [
    "deepfake artificial face", "robotic manipulated face", "blurry distorted deepfake", 
    "AI generated face", "synthetic human face", "face swap artifact", 
    "unnatural skin texture", "glitched face boundaries", "inconsistent lighting on face", 
    "wax-like artificial skin", "mismatched skin tone", "unnaturally smooth face", 
    "computer generated portrait", "manipulated video frame", "deep learning generated face"
]

# --- 3. LOAD MODELS ---
print("Loading Models (HF ViT + OpenCLIP)...")

model_id = "dima806/deepfake_vs_real_image_detection"
# Adding token fetch for private/gated models if needed
processor = AutoImageProcessor.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
hf_model = AutoModelForImageClassification.from_pretrained(model_id, token=os.environ.get("HF_TOKEN")).to(DEVICE)
if DEVICE == "cuda": hf_model = hf_model.half()
hf_model.eval()

clip_model, _, clip_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

with torch.no_grad():
    real_tokens = tokenizer(REAL_PROMPTS).to(DEVICE)
    fake_tokens = tokenizer(FAKE_PROMPTS).to(DEVICE)
    real_text_embs = F.normalize(clip_model.encode_text(real_tokens), dim=-1)
    fake_text_embs = F.normalize(clip_model.encode_text(fake_tokens), dim=-1)

print("✅ All Models Loaded")

# --- 4. UTILS & MATH ---
def extract_faces_retina(frame_img):
    # SPEED OPTIMIZATION: Shrink image before passing to RetinaFace
    frame_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    frame_np = np.array(frame_img)
    try: faces = RetinaFace.detect_faces(frame_np)
    except: return frame_img

    if not faces or isinstance(faces, tuple): return frame_img

    max_area = 0; best_face = frame_img
    for key in faces:
        identity = faces[key]
        x1, y1, x2, y2 = identity["facial_area"]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            margin = int((x2 - x1) * 0.2)
            best_face = frame_img.crop((max(0, x1 - margin), max(0, y1 - margin), 
                                        min(frame_np.shape[1], x2 + margin), 
                                        min(frame_np.shape[0], y2 + margin)))
    return best_face

def extract_10_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: total = 10 
    indices = [int(i * (total - 1) / 9) for i in range(10)]
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames

def get_clip_similarity(face_img):
    img_tensor = clip_transform(face_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = F.normalize(clip_model.encode_image(img_tensor), dim=-1)
        sim_real = (img_emb @ real_text_embs.T).mean().item()
        sim_fake = (img_emb @ fake_text_embs.T).mean().item()
    return sim_real, sim_fake

# --- 5. CLAUDE 3.5 SONNET ---
def get_claude_reasoning(image, fake_votes, real_votes, sim_real, sim_fake, verdict):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    prompt_text = (
        f"Role: Forensic Video Analyst.\n"
        f"Data Analysis:\n"
        f"- Frame Voting System: {fake_votes}/10 frames detected as FAKE.\n"
        f"- Mathematical Semantic Similarity (OpenCLIP): FAKE correlation = {sim_fake:.4f}, REAL correlation = {sim_real:.4f}.\n"
        f"- Final System Verdict: {verdict}\n\n"
        f"Instruction:\n"
        f"Based on the data and the provided frame image, write EXACTLY 2 lines of reasoning explaining why this video is {verdict}. "
        f"Keep it highly technical, concise, and focused on visual artifacts or semantic scores. Do not exceed 2 sentences."
    )

    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
        ]}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                               json=payload)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Reasoning unavailable due to API error."

# --- 6. MAIN PIPELINE ---
def analyze_video(video, progress=gr.Progress()):
    clear_memory()
    if video is None: return None, "No video uploaded.", "Error"

    progress(0, desc="Extracting frames from video...")
    frames = extract_10_frames(video)
    if not frames: return None, "Could not extract frames.", "Error"

    fake_votes = 0
    real_votes = 0
    total_sim_real = 0
    total_sim_fake = 0
    processed_faces = []
    best_face_for_claude = None
    max_fake_conf = 0

    print("🔍 Analyzing 10 frames...")
    for i, frame in enumerate(frames):
        progress((i + 1) / 10, desc=f"Analyzing Frame {i + 1} of 10...")
        
        face = extract_faces_retina(frame)
        processed_faces.append(face)

        sim_r, sim_f = get_clip_similarity(face)
        total_sim_real += sim_r
        total_sim_fake += sim_f

        with torch.no_grad():
            inputs = processor(images=face, return_tensors="pt").to(DEVICE)
            if DEVICE == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float else v for k, v in inputs.items()}

            outputs = hf_model(**inputs)
            logits = outputs.logits
            pred_idx = logits.argmax(-1).item()
            label = hf_model.config.id2label[pred_idx].lower()
            confidence = torch.softmax(logits, dim=1)[0][pred_idx].item() * 100

            if "fake" in label and confidence > 75.0:
                fake_votes += 1
                if confidence > max_fake_conf:
                    max_fake_conf = confidence
                    best_face_for_claude = face
            else:
                real_votes += 1

    if best_face_for_claude is None:
        best_face_for_claude = processed_faces[0]

    progress(0.95, desc="Writing Forensic Report with Claude...")
    avg_sim_real = total_sim_real / 10
    avg_sim_fake = total_sim_fake / 10
    is_forged = fake_votes >= 6
    final_verdict = "FORGED" if is_forged else "REAL"

    claude_reasoning = get_claude_reasoning(
        best_face_for_claude, fake_votes, real_votes, avg_sim_real, avg_sim_fake, final_verdict
    )

    color = "red" if is_forged else "green"
    verdict_text = "🚨 FORGED (FAKE VIDEO)" if is_forged else "✅ REAL VIDEO"

    stats_summary = (
        f"## Verdict: <span style='color:{color}'>{verdict_text}</span>\n"
        f"**Voting System (10 Frames):** {fake_votes} Fake / {real_votes} Real\n\n"
        f"**OpenCLIP Semantic Similarity:**\n"
        f"- Correlation with Fake: `{avg_sim_fake:.4f}`\n"
        f"- Correlation with Real: `{avg_sim_real:.4f}`"
    )

    # Return EXACTLY 4 items to match your app.py outputs list
    return (
        best_face_for_claude,  # 1. Goes to vid_img_out (Image)
        stats_summary,         # 2. Goes to vid_txt_out (Markdown)
        claude_reasoning,      # 3. Goes to vid_reason_out (Textbox)
        verdict_text           # 4. Goes to vid_hidden (Textbox)
    )
# Removed standalone UI logic because app.py handles it.