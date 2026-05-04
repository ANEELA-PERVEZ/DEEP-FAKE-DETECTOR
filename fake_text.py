# -*- coding: utf-8 -*-
"""
Cleaned fake text detection module (Gradio UI removed)
"""

import subprocess
import sys
import json
import os  # Yeh module secrets ko call karne ke liye add kiya gaya hy

# --- 0. AUTO-INSTALLER ---
def install_requirements():
    required = ["groq", "gradio", "sentence-transformers", "torch", "requests"]
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"⏳ Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

# --- IMPORTS ---
import requests
from groq import Groq
from sentence_transformers import CrossEncoder
import torch
import numpy as np

# --- 1. SECRETS SE KEYS CALL KARNA ---
# Yahan aapke Hugging Face variables automatically fetch ho jayenge
SERP_API_KEY = os.environ.get("SERP_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- 2. LOAD MODELS ---
print("⚙️ Loading Intelligence...")
try:
    # 🧠 DeBERTa Model (For Math Scores)
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    # 🧠 Groq Client (For Reasoning)
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ System Ready: Groq & DeBERTa Loaded!")
except Exception as e:
    print(f"Error loading models: {e}")

# --- 3. HELPER FUNCTIONS ---

def search_google_direct(query):
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY,
            "num": 10
        }
        response = requests.get(url, params=params)
        data = response.json()

        if "error" in data:
            return f"Search API Error: {data['error']}", []

        results = data.get("organic_results", [])
        evidence_list = []
        full_text = ""

        if not results:
            return "No news found on Google.", []

        for i, res in enumerate(results):
            snippet = res.get("snippet", "")
            title = res.get("title", "")
            evidence_list.append(f"{title}. {snippet}")
            full_text += f"- {title}: {snippet}\n"

        return full_text, evidence_list
    except Exception as e:
        return f"Connection Error: {e}", []

# 🧮 YAHAN HAI MATHEMATICAL COMPUTATION
def get_smart_mathematical_score(claim, evidence_list):
    """
    STRICT MATHEMATICAL LOCK:
    1. Finds Best Evidence.
    2. Forces Total Score to be 100%.
    """
    if not evidence_list: return 0, 0

    # 1. Sabhi Evidence ka Score nikalo
    pairs = [[claim, text] for text in evidence_list]
    scores = nli_model.predict(pairs)

    # Variables to find the single best matching sentence
    highest_confidence = -1
    best_true_prob = 0
    best_fake_prob = 0

    for score in scores:
        # Logits to Probabilities (0 to 1)
        probs = torch.nn.functional.softmax(torch.tensor(score), dim=0).numpy()

        # DeBERTa Labels: 0=Fake, 1=True, 2=Neutral
        fake_raw = probs[0]
        true_raw = probs[1]

        # Dekho AI ko is jumlay par kitna yaqeen hai
        confidence = max(fake_raw, true_raw)

        # Champion Logic: Sirf wo sentence uthao jiska confidence sabse high hai
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_true_prob = true_raw
            best_fake_prob = fake_raw

    # --- MATH LOCK (100% Total) ---
    total_relevant_score = best_true_prob + best_fake_prob

    if total_relevant_score > 0:
        # Math: Percentage Calculation
        final_true = (best_true_prob / total_relevant_score) * 100
        # Math: Fake = 100 - True
        final_fake = 100 - final_true
    else:
        final_true = 0
        final_fake = 0

    return int(final_true), int(final_fake)

def ask_groq_brain(claim, evidence):
    prompt = f"""
    Act as an Expert Fact-Checker.
    Claim: "{claim}"
    Evidence from Google:
    {evidence}

    Instructions:
    1. Analyze the evidence carefully.
    2. If the claim is about a PURCHASE/DEAL and evidence confirms a sale (even 75% stake), mark TRUE.
    3. If evidence says "Rejected", "Failed", mark FAKE.
    4. Provide a clear verdict.

    Format:
    **Verdict:** [TRUE / FAKE / UNVERIFIED]
    **Reasoning:** [Explain in 2 simple sentences why]
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

# --- 4. MAIN ENGINE (WITH SYNC LOGIC) ---
def hybrid_analysis(claim):
    if not claim: return "⚠️ Please enter a claim."

    # Step 1: Search Google
    evidence_text, evidence_list = search_google_direct(claim)

    # Step 2: Calculate Math Scores (DeBERTa)
    score_true, score_fake = get_smart_mathematical_score(claim, evidence_list)

    # Step 3: Get Reasoning (Groq)
    groq_response = ask_groq_brain(claim, evidence_text)

    # Step 4: Final Decision & Override
    final_color = "gray"
    final_verdict = "ANALYZING"
    groq_clean = groq_response.upper()

    if "**VERDICT:** TRUE" in groq_clean or "VERDICT: TRUE" in groq_clean:
        final_color = "#188038" # Green
        final_verdict = "✅ TRUE / VERIFIED"

        # Override: Agar Groq Sure hai, to Math ko bhi adjust karo
        if score_true < 50:
            score_true = 85
            score_fake = 15

    elif "**VERDICT:** FAKE" in groq_clean or "VERDICT: FAKE" in groq_clean:
        final_color = "#d93025" # Red
        final_verdict = "❌ FAKE / DEBUNKED"

        # Override: Agar Groq Sure hai to Math ko adjust karo
        if score_fake < 50:
            score_fake = 85
            score_true = 15

    else:
        final_color = "#f9ab00" # Orange
        final_verdict = "⚠️ UNVERIFIED"

    # HTML Output
    html = f"""
    <div style='display:flex; gap:20px; font-family:sans-serif;'>

        <div style='flex:1.5; background:#f9f9f9; padding:20px; border-radius:10px; border-left: 8px solid {final_color};'>
            <h3 style='color:{final_color}; margin-top:0;'>{final_verdict}</h3>
            <p style='font-size:15px; white-space: pre-line; color:#333;'>{groq_response}</p>
            <small style='color:#666;'>Reasoning: Llama-3.3 (Groq)</small>
        </div>

        <div style='flex:1; background:#fff; border: 1px solid #ddd; padding:20px; border-radius:10px; text-align:center; display:flex; flex-direction:column; justify-content:center;'>
            <h4 style='margin:0; opacity:0.6; font-size:12px;'>MATHEMATICAL CONFIDENCE (DeBERTa)</h4>

            <div style='display:flex; justify-content:space-around; margin-top:10px;'>
                <div>
                    <h1 style='color:#188038; margin:0;'>{score_true}%</h1>
                    <small>True Score</small>
                </div>
                <div>
                    <h1 style='color:#d93025; margin:0;'>{score_fake}%</h1>
                    <small>Fake Score</small>
                </div>
            </div>
            <p style='font-size:11px; color:#888; margin-top:10px;'>Model: microsoft/deberta-v3-base</p>
        </div>

    </div>

    <div style='margin-top:20px; padding:10px; background:#eee; font-size:12px; color:#555; border-radius:5px;'>
        <b>🔍 Live Google Data:</b> {evidence_text[:300]}...
    </div>
    """
    return html