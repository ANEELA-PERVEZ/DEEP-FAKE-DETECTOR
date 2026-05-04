# -*- coding: utf-8 -*-
"""
Cleaned fake vs forged vs real audio module (Gradio UI & Notebook Magics removed)
Fixed for PyTorch 2.6+ compatibility with LAION-CLAP checkpoints
"""

import os
import urllib.request
from huggingface_hub import login
import torch
import torch.serialization
import librosa
import numpy as np
import numpy
import laion_clap
import whisper
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoModelForAudioFrameClassification
from groq import Groq

# --- FETCH SECRETS & LOGIN ---
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✅ Login Successful! Ab aap models load kar sakti hain.")
else:
    print("⚠️ WARNING: HF_TOKEN not found in Secrets. Private models load nahi honge.")

# --- SETUP & CONFIG ---
client = Groq(api_key=GROQ_API_KEY)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AST_MODEL_PATH = "aneela-pervez/FAKE-AUDIO"

# --- CATEGORIZED FORENSIC LABELS ---
FORENSIC_CATEGORIES = {
    "AUTHENTIC": ["Natural human speech", "Human voice with natural breathing", "Consistent room ambiance"],
    "AI_SYNTHETIC": ["AI voice clone robotic smoothness", "Synthetic neural vocoder", "Deepfake metallic resonance"],
    "MANIPULATED": ["Manually spliced audio", "Inconsistent stitching artifacts", "Micro-cuts and clicks"],
    "DISGUISED": ["Artificial pitch shifting", "Muffled voice mask identity", "Time-stretched artifacts"]
}
ALL_LABELS = [item for sublist in FORENSIC_CATEGORIES.values() for item in sublist]

# --- MODEL LOADING ---
print("🔄 Initializing Fuzzy Forensic Engine v3.9...")
whisper_model = whisper.load_model("base", device=DEVICE)
id_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
id_model = AutoModelForAudioFrameClassification.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE).eval()

try:
    ast_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_PATH)
    ast_model = AutoModelForAudioClassification.from_pretrained(AST_MODEL_PATH).to(DEVICE).eval()
except Exception as e:
    print(f"⚠️ AST Model issue (using default logic if fails): {e}")

# --- CLAP MODEL LOADING (PyTorch 2.6+ Full Fix) ---
# Step 1: Checkpoint pehle local mein download karein
CLAP_CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_PATH = "/tmp/clap_music_audioset.pt"

if not os.path.exists(CLAP_CKPT_PATH):
    print("Downloading CLAP checkpoint to local disk...")
    urllib.request.urlretrieve(CLAP_CKPT_URL, CLAP_CKPT_PATH)
    print("✅ CLAP checkpoint downloaded!")
else:
    print("✅ CLAP checkpoint already cached.")

# Step 2: PyTorch 2.6+ safe globals fix
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# Step 3: CLAP model initialize karein
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny').to(DEVICE)

# Step 4: Manually load checkpoint with strict=False
# strict=False: unexpected/missing keys jaise position_ids ko ignore karta hai
print("🔄 Loading CLAP checkpoint (strict=False)...")
ckpt = torch.load(CLAP_CKPT_PATH, map_location=DEVICE, weights_only=False)

# Checkpoint structure check
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

clap_model.model.load_state_dict(state_dict, strict=False)
clap_model.eval()
print("✅ CLAP model loaded successfully!")


def extract_dna_embeddings(y):
    y = np.ascontiguousarray(y)
    inputs = id_extractor(y, sampling_rate=16000, return_tensors="pt")
    inputs = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)).to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = id_model(**inputs).logits
        emb = torch.mean(logits, dim=1).cpu().numpy()
    return normalize(emb)


def analyze_voice_forensics(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        y_48, _ = librosa.load(audio_path, sr=48000)
        y = y / (np.max(np.abs(y)) + 1e-9)
        y = np.ascontiguousarray(y)

        # 1. SEGMENT-WISE CROSS VALIDATION
        segments = np.array_split(y, 3)
        seg_embs = [extract_dna_embeddings(s) for s in segments]
        cos_sim = cosine_similarity(seg_embs[0], seg_embs[-1])[0][0]
        euc_dist = euclidean_distances(seg_embs[0], seg_embs[-1])[0][0]

        # 2. FEATURE EXTRACTION
        transcription = whisper_model.transcribe(audio_path)["text"]
        inputs = ast_extractor(y, sampling_rate=16000, return_tensors="pt")
        inputs = {k: (v.clone().detach() if isinstance(v, torch.Tensor) else torch.tensor(v)).to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            ast_prob = torch.nn.functional.softmax(ast_model(**inputs).logits, dim=-1).cpu().numpy()[0][1]
            ast_score = ast_prob * 100

        # CLAP EMBEDDINGS
        audio_emb_np = clap_model.get_audio_embedding_from_data(x=[y_48])
        text_emb_np = clap_model.get_text_embedding(ALL_LABELS)
        audio_emb = torch.from_numpy(audio_emb_np).to(DEVICE)
        text_emb = torch.from_numpy(text_emb_np).to(DEVICE)
        c_sims = torch.nn.functional.cosine_similarity(audio_emb, text_emb)
        top_idx = torch.argmax(c_sims).item()
        clap_label = ALL_LABELS[top_idx]

        category = "UNKNOWN"
        for cat, labels in FORENSIC_CATEGORIES.items():
            if clap_label in labels:
                category = cat
                break

        num_edits = len(np.where(np.diff(librosa.onset.onset_strength(y=y, sr=16000)) > 6.5)[0])

        # FUZZY LOGIC MEMBERSHIP SCORING
        mu_ai = min(1.0, max(0.0, (ast_score - 40) / 45))
        mu_spliced = min(1.0, max(0.0, (num_edits - 8) / 10))
        dna_conf = (cos_sim - 0.85) / 0.10
        cat_conf = 1.0 if category == "AUTHENTIC" else 0.0
        mu_auth = min(1.0, max(0.0, (dna_conf + cat_conf) / 2))

        # FUZZY DECISION ENGINE (V4.0)
        if mu_auth > 0.80 and mu_spliced < 0.75:
            verdict = "✅ AUTHENTIC HUMAN VOICE"
            category_override = "AUTHENTIC"
        elif mu_ai > 0.85 and mu_auth < 0.80:
            verdict = "🚨 AI CLONE DETECTED"
            category_override = "AI_SYNTHETIC"
        else:
            verdict = "⚠️ MANIPULATED REAL VOICE (Spliced)"
            category_override = "MANIPULATED"

        # LLM REASONING
        audit_context = (f"Verdict: {verdict}. Category: {category_override}. Metrics: ID {cos_sim:.2f}, "
                         f"Texture {ast_score:.1f}%, Cuts {num_edits}. "
                         f"Fuzzy Scores: AI={mu_ai:.2f}, Spliced={mu_spliced:.2f}, Auth={mu_auth:.2f}")

        prompt = (f"Act as a Senior Forensic Auditor. {audit_context} "
                  "Instruction: Justify the verdict. If identity stability is high (Auth > 0.8), "
                  "emphasize that DNA integrity confirms human origin despite high spectral texture. "
                  "Give a 2-line direct forensic conclusion.")

        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        reasoning = res.choices[0].message.content

        evidence = f"Category: {category} | ID: {cos_sim:.2f} | Texture: {ast_score:.1f}% | Cuts: {num_edits}"
        return verdict, evidence, reasoning

    except Exception as e:
        return "Error", str(e), "N/A"