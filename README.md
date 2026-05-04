---
title: DEEP FAKE DETECTOR

short_description: Multimodal deepfake detection across 5 modalities
---

# Multimodal Deepfake Detection System

A modular AI system for detecting fake content across 5 modalities — text, audio, image, video, and lip-sync — using hybrid CNN-Transformer-LLM architectures.

## Live Demo

Try the live system using the tabs above:

- **Text Verification** — Detects fake news using SERP API + DeBERTa + Llama-3.3
- **Audio Forensics** — Identifies AI-cloned voices using Wav2Vec2 + AST + CLAP
- **Image Analysis** — Detects deepfake images using ResNet-18 + Vision Transformer
- **Video Forensics** — Analyzes video deepfakes with multi-frame voting + OpenCLIP
- **Multi-Modal Lip-Sync** — Detects audio-video mismatch using LSE-Net + Claude reasoning

## Performance

| Modality | Accuracy |
|----------|----------|
| Text     | 99%      |
| Image    | 92%      |
| Audio    | 91%      |
| Video    | 91%      |
| Lip-Sync | 89%      |

## Tech Stack

- **Frameworks:** PyTorch, Hugging Face Transformers, Gradio
- **Vision Models:** Vision Transformer (ViT), ResNet-18, OpenCLIP
- **Language Models:** DeBERTa-v3, Llama-3.3, Claude 3.5 Sonnet
- **Audio Models:** Wav2Vec2, AST, CLAP, Whisper
- **Datasets:** FaceForensics++, FakeAVCeleb, ASVspoof

## About

This system was developed as part of MS thesis research at GIFT University, Pakistan.

**Author:** Aneela Pervez  
**Institution:** GIFT University, Department of Computer Science  
**Supervisor:** Dr. Muhammad Ziad Nayyer

## Links

- **GitHub Repository:** https://github.com/ANEELA-PERVEZ
- **LinkedIn:** https://linkedin.com/in/aneela-pervez-95472b310
- **Paper:** Coming soon on arXiv

## Citation

If you use this system in your research, please cite:

```
