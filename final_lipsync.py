# -*- coding: utf-8 -*-
import os
import gc
import urllib.request
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import librosa
import numpy as np
import base64
import requests
from io import BytesIO
from PIL import Image
import gradio as gr

# Foundation Models
import whisper
import open_clip
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import laion_clap
from retinaface import RetinaFace
from huggingface_hub import hf_hub_download

# --- 🚨 MASTER FIX FOR PYTORCH 2.6 SECURITY ---
import torch.serialization
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass

# Robust torch.load patch — handles both positional and keyword args
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False  # Always force weights_only=False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
# ----------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# ==========================================
# 1. 15 EXACT PROMPTS FOR EACH MODALITY
# ==========================================
PROMPTS = {
    "REAL_VIDEO": [
        "a natural photograph of a real human face", "authentic human face with natural skin pores",
        "real video frame with consistent shadows", "organic micro-expressions and facial muscle movement",
        "genuine eye reflection and natural pupil dilation", "natural hair strands and authentic hairline",
        "unedited facial texture with natural biological symmetry", "real camera photograph without digital manipulation",
        "natural depth of field and authentic background blur", "real skin tone variations and biological flushing",
        "authentic lip movement matching natural speech", "raw unedited human portrait recording",
        "natural lighting interacting with real human skin", "authentic biological features without smoothing",
        "real-world video recording of a genuine person"
    ],
    "FAKE_VIDEO": [
        "an AI generated deepfake face", "synthetic face with digital artifacts and wax-like skin",
        "face swap with morphed features and edge seams", "inconsistent lighting and artificial CGI shadows",
        "unnatural eye blinking and robotic gaze", "resolution mismatch between face and background",
        "GAN generated artifacts around teeth and mouth", "diffusion model synthetic rendering errors",
        "blurred chin boundaries and artificial jawline blending", "robotic facial stiffness and lack of micro-expressions",
        "artificial skin smoothing and plastic-like texture", "deepfake face replacement glitches",
        "unnatural rendering of hair and temporal flickering", "synthetic eye catchlights and flat rendering",
        "AI deepfake avatar with uncanny valley effect"
    ],
    "REAL_AUDIO": [
        "clear natural human conversation", "authentic human voice with natural breathing",
        "real speech with consistent room ambiance", "organic vocal cord resonance and natural pitch",
        "genuine emotional vocal tone variations", "natural speech hesitations and micro-pauses",
        "real unedited microphone recording", "biological human breathing between sentences",
        "authentic pronunciation nuances and dialects", "real-time spontaneous speaking rhythm",
        "natural interaction with background acoustic environment", "unaltered biological vocal frequencies",
        "genuine human dialogue with natural dynamics", "real human speech without digital artifacts",
        "authentic voice recording of a living person"
    ],
    "FAKE_AUDIO": [
        "robotic AI generated voice clone", "monotone synthetic deepfake speech without emotion",
        "artificial voice with metallic vocoder resonance", "AI text-to-speech rendering artifacts",
        "unnatural breathing gaps and spliced audio breaths", "digital audio cloning glitches and clicks",
        "synthetic pitch shifting and algorithmic tuning", "unnatural emotional emotional flatness in speech",
        "deepfake voice generator output with muffled artifacts", "inconsistent background noise transitions",
        "artificial neural network speech synthesis", "disjointed phoneme stitching in audio track",
        "unnatural pronunciation of complex syllables", "synthesized AI audio lacking acoustic depth",
        "AI voice clone with uncanny mechanical overtone"
    ]
}

# ==========================================
# 2. EXACT MODEL ARCHITECTURES
# ==========================================
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout))
        self.act = nn.ReLU()
        self.residual = residual
    def forward(self, x):
        out = self.conv_block(x)
        if self.residual: out += x
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()
        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, 7, 1, 3), Conv2d(32, 64, 5, (1, 2), 1), Conv2d(64, 64, 3, 1, 1, True), Conv2d(64, 64, 3, 1, 1, True),
            Conv2d(64, 128, 3, 2, 1), Conv2d(128, 128, 3, 1, 1, True), Conv2d(128, 128, 3, 1, 1, True), Conv2d(128, 128, 3, 1, 1, True),
            Conv2d(128, 256, 3, 2, 1), Conv2d(256, 256, 3, 1, 1, True), Conv2d(256, 256, 3, 1, 1, True), Conv2d(256, 512, 3, 2, 1),
            Conv2d(512, 512, 3, 1, 1, True), Conv2d(512, 512, 3, 1, 1, True), Conv2d(512, 512, 3, 2, 1), Conv2d(512, 512, 3, 1, 0),
            Conv2d(512, 512, 1, 1, 0),
        )
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1), Conv2d(32, 32, 3, 1, 1, True), Conv2d(32, 32, 3, 1, 1, True), Conv2d(32, 64, 3, (3, 1), 1),
            Conv2d(64, 64, 3, 1, 1, True), Conv2d(64, 64, 3, 1, 1, True), Conv2d(64, 128, 3, 3, 1), Conv2d(128, 128, 3, 1, 1, True),
            Conv2d(128, 128, 3, 1, 1, True), Conv2d(128, 256, 3, (3, 2), 1), Conv2d(256, 256, 3, 1, 1, True), Conv2d(256, 256, 3, 1, 1, True),
            Conv2d(256, 512, 3, 1, 0), Conv2d(512, 512, 1, 1, 0),
        )
    def forward(self, aud, fac):
        return F.normalize(self.audio_encoder(aud).view(aud.size(0), -1), p=2, dim=1), F.normalize(self.face_encoder(fac).view(fac.size(0), -1), p=2, dim=1)

class FusionLipSyncModel(nn.Module):
    def __init__(self):
        super().__init__()
        clip_m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=False)
        self.visual_branch = nn.Module()
        self.visual_branch.clip_model = clip_m.visual
        self.audio_branch = nn.Module()
        self.audio_branch.whisper = whisper.load_model("base").encoder
        self.audio_branch.classifier = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
        self.fusion_module = nn.Module()
        self.fusion_module.conv_visual = nn.Conv1d(768, 512, kernel_size=1)
        self.fusion_module.conv_audio = nn.Conv1d(512, 512, kernel_size=1)
        self.fusion_module.classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 2))

    def forward(self, mel, img):
        v_feat = self.visual_branch.clip_model.conv1(img)
        v_feat = v_feat.reshape(v_feat.shape[0], v_feat.shape[1], -1).permute(0, 2, 1)
        v_feat = torch.cat([self.visual_branch.clip_model.class_embedding.to(v_feat.dtype) + torch.zeros(v_feat.shape[0], 1, v_feat.shape[-1], dtype=v_feat.dtype, device=v_feat.device), v_feat], dim=1)
        v_feat = v_feat + self.visual_branch.clip_model.positional_embedding.to(v_feat.dtype)
        v_feat = self.visual_branch.clip_model.ln_pre(v_feat).permute(1, 0, 2)
        v_feat = self.visual_branch.clip_model.transformer(v_feat).permute(1, 0, 2)
        v_feat = v_feat[:, 0, :]

        a_feat = self.audio_branch.whisper(mel).mean(dim=1)
        fused = torch.cat([self.fusion_module.conv_visual(v_feat.unsqueeze(-1)).squeeze(-1),
                           self.fusion_module.conv_audio(a_feat.unsqueeze(-1)).squeeze(-1)], dim=1)
        return torch.softmax(self.fusion_module.classifier(fused), dim=1)

class MyFakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 2)
    def forward(self, x): return torch.softmax(self.fc(x.view(x.size(0), -1)), dim=1)


# ==========================================
# 3. CLAP CHECKPOINT — DOWNLOAD HELPER
# ==========================================
def load_clap_safely(clap_module, device):
    """
    Load CLAP checkpoint with PyTorch 2.6+ compatibility.
    Downloads checkpoint locally and uses strict=False to handle key mismatches.
    """
    CLAP_CKPT_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"
    CLAP_CKPT_PATH = "/tmp/clap_music_audioset.pt"

    # Step 1: Download checkpoint locally if not cached
    if not os.path.exists(CLAP_CKPT_PATH):
        print("⬇️ Downloading CLAP checkpoint to local disk...")
        urllib.request.urlretrieve(CLAP_CKPT_URL, CLAP_CKPT_PATH)
        print("✅ CLAP checkpoint downloaded!")
    else:
        print("✅ CLAP checkpoint already cached.")

    # Step 2: Load checkpoint with weights_only=False
    print("🔄 Loading CLAP checkpoint (strict=False)...")
    ckpt = torch.load(CLAP_CKPT_PATH, map_location=device)

    # Step 3: Extract state_dict from checkpoint
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Step 4: Load with strict=False to ignore unexpected/missing keys
    clap_module.model.load_state_dict(state_dict, strict=False)
    print("✅ CLAP model loaded successfully in lipsync!")


# ==========================================
# 4. LOAD ALL MODELS (REDIRECTION TO HUB)
# ==========================================
def load_all_models():
    print("Loading Models...")
    m = {}

    clip_m, _, clip_p = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    m["clip"], m["clip_p"] = clip_m.to(DEVICE).eval(), clip_p

    m["ast_ext"] = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    m["ast_mod"] = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(DEVICE).eval()

    # CLAP loading with PyTorch 2.6+ fix
    m["clap"] = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny').to(DEVICE)
    load_clap_safely(m["clap"], DEVICE)

    sync_m = SyncNet_color().to(DEVICE)
    try:
        sync_m.load_state_dict(torch.load(hf_hub_download(repo_id="camenduru/Wav2Lip", filename="syncnet_v2.pth"), map_location=DEVICE)['state_dict'], strict=False)
    except Exception as e:
        print(f"⚠️ SyncNet load issue: {e}")
    m["hf_sync"] = sync_m.eval()

    # Step 8 Models - Downloading from your HuggingFace Repo
    img_m = MyFakeImageModel().to(DEVICE)
    try:
        img_path = hf_hub_download(repo_id="aneela-pervez/My-Deepfake-Models", filename="checkpoint_step000080000.pth")
        img_m.load_state_dict(torch.load(img_path, map_location=DEVICE), strict=False)
    except Exception as e:
        print(f"⚠️ Custom image model load issue: {e}")
    m["custom_img"] = img_m.eval()

    fusion_m = FusionLipSyncModel().to(DEVICE)
    try:
        fusion_path = hf_hub_download(repo_id="aneela-pervez/My-Deepfake-Models", filename="best_model.pth")
        fusion_state = torch.load(fusion_path, map_location=DEVICE)
        if isinstance(fusion_state, dict) and 'model_state_dict' in fusion_state:
            fusion_state = fusion_state['model_state_dict']
        fusion_m.load_state_dict(fusion_state, strict=False)
    except Exception as e:
        print(f"⚠️ Fusion model load issue: {e}")
    m["custom_fusion"] = fusion_m.eval()

    return m

models = load_all_models()


# ==========================================
# 5. ANALYSIS LOGIC (UNCHANGED)
# ==========================================
def extract_mouth(frame_np):
    try: faces = RetinaFace.detect_faces(frame_np)
    except: return None
    if not faces or isinstance(faces, tuple): return None
    best_box = max([faces[k]["facial_area"] for k in faces], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1, y1, x2, y2 = best_box
    m_crop = frame_np[y1 + int((y2-y1)*0.5):y2, x1:x2]
    return cv2.resize(m_crop, (96, 48)) if m_crop.size > 0 else None

def full_analysis(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    # Fast Processing Logic (Har 5th frame use karega taake GPU pe time bache)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_count % 5 == 0:
            temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(temp_frame, (640, 360))
            frames.append(small_frame)

        frame_count += 1
        if len(frames) > 150: break
    cap.release()

    if len(frames) == 0: return None, 0, 0, 0, 0, 0, 0, 0, 0, 0

    mid_frame = Image.fromarray(frames[len(frames)//2])
    face_tensor = models["clip_p"](mid_frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        i_feat = F.normalize(models["clip"].encode_image(face_tensor), dim=-1)
        t_feat = F.normalize(models["clip"].encode_text(open_clip.tokenize(PROMPTS["REAL_VIDEO"] + PROMPTS["FAKE_VIDEO"]).to(DEVICE)), dim=-1)
        clip_probs = (100.0 * i_feat @ t_feat.T).softmax(dim=-1)
        clip_v_score = clip_probs[0][15:].sum().item() * 100
        try: custom_v_score = models["custom_img"](i_feat)[0][1].item() * 100
        except: custom_v_score = clip_v_score
    final_v_score = (clip_v_score * 0.7) + (custom_v_score * 0.3)

    # Fast Audio Loading Logic
    y, sr = librosa.load(video_path, sr=48000, duration=10)
    y_16, _ = librosa.load(video_path, sr=16000, duration=10)

    with torch.no_grad():
        a_emb = torch.from_numpy(models["clap"].get_audio_embedding_from_data(x=[y])).to(DEVICE)
        r_a_emb = torch.from_numpy(models["clap"].get_text_embedding(PROMPTS["REAL_AUDIO"])).to(DEVICE)
        f_a_emb = torch.from_numpy(models["clap"].get_text_embedding(PROMPTS["FAKE_AUDIO"])).to(DEVICE)
        s_r = F.cosine_similarity(a_emb, r_a_emb).mean().item()
        s_f = F.cosine_similarity(a_emb, f_a_emb).mean().item()
        clap_a_score = (max(0, s_f) / (max(0, s_r) + max(0, s_f) + 1e-5)) * 100
        ast_a_score = torch.softmax(models["ast_mod"](**models["ast_ext"](y_16, sampling_rate=16000, return_tensors="pt").to(DEVICE)).logits, dim=-1)[0][1].item() * 100
    final_a_score = (clap_a_score * 0.75) + (ast_a_score * 0.25)

    mel_16k = librosa.power_to_db(librosa.feature.melspectrogram(y=y_16, sr=16000, n_fft=800, hop_length=200, n_mels=80), ref=np.max)
    hf_scores = []
    for i in range(0, len(frames) - 5, 3):
        mouths = [extract_mouth(f) for f in frames[i:i+5]]
        if any(m is None for m in mouths): continue
        f_t = torch.FloatTensor(np.concatenate([m.transpose((2,0,1))/255.0 for m in mouths], axis=0)).unsqueeze(0).to(DEVICE)
        s_i = int((i/fps)*(16000/200))
        m_c = mel_16k[:, s_i:s_i+16]
        if m_c.shape[1] < 16: m_c = np.pad(m_c, ((0,0), (0, 16 - m_c.shape[1])))
        m_t = torch.FloatTensor(m_c[:, :16]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            a_e, v_e = models["hf_sync"](m_t, f_t)
            hf_scores.append(F.cosine_similarity(a_e, v_e).item())
    hf_sync_anomaly = (1.0 - max(0, np.mean(hf_scores))) * 100.0 if hf_scores else 100.0
    try:
        aud_w = whisper.pad_or_trim(whisper.load_audio(video_path))
        mel_w = whisper.log_mel_spectrogram(aud_w).to(DEVICE).unsqueeze(0)
        with torch.no_grad(): custom_fusion_score = models["custom_fusion"](mel_w, face_tensor)[0][1].item() * 100
    except: custom_fusion_score = hf_sync_anomaly
    final_sync_score = (hf_sync_anomaly * 0.6) + (custom_fusion_score * 0.4)

    return mid_frame, final_v_score, final_a_score, final_sync_score, clip_v_score, custom_v_score, clap_a_score, ast_a_score, hf_sync_anomaly, custom_fusion_score

def master_pipeline(video_path):
    if not video_path: return None, "No video provided.", "Error"
    res = full_analysis(video_path)
    if res is None or res[0] is None:
        return None, "Analysis failed to process video.", "Error"

    img, v_score, a_score, sync_score, clip_v, cust_v, clap_a, ast_a, hf_s, cust_s = res

    is_v_fake, is_a_fake, is_sync_bad = v_score >= 50, a_score >= 50, sync_score >= 60
    if is_v_fake and is_a_fake: case = "CASE 1: Full Deep Fake"
    elif is_v_fake: case = "CASE 2: Fake Video + Real Audio"
    elif is_a_fake: case = "CASE 3: Real Video + Fake Audio"
    elif is_sync_bad: case = "CASE 4: Lip Sync Issue"
    else: case = "CASE 5: Full Authentic"

    prompt_t = f"Analyze scores and confirm {case}. Give 2 lines simple reasoning. Scores: Vision {v_score:.1f}%, Audio {a_score:.1f}%, Sync {sync_score:.1f}%"
    buf = BytesIO(); img.save(buf, format="JPEG")
    try:
        req = requests.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                            json={"model": "anthropic/claude-3.5-sonnet", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt_t}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"}}]}]})
        reason = req.json()['choices'][0]['message']['content']
    except: reason = "Reasoning unavailable."

    summary = f"## 📂 {case}\n\nVision: {v_score:.1f}% | Audio: {a_score:.1f}% | Sync: {sync_score:.1f}%"
    return img, summary, reason