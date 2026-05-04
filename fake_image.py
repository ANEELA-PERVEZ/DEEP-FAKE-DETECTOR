# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from huggingface_hub import hf_hub_download

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

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Model Architecture (SOTA Hybrid Network)
class Deepfake_Hybrid_Network(nn.Module):
    def __init__(self, num_classes=2):
        super(Deepfake_Hybrid_Network, self).__init__()
        resnet = models.resnet18(weights=None)
        self.cnn_extractor = nn.Sequential(*list(resnet.children())[:-2]) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn_extractor(x) 
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        return self.classifier(x)

# 3. Load Model from Hugging Face Hub
print("Loading Image Hybrid SOTA Model...")
model = Deepfake_Hybrid_Network(num_classes=2).to(device)

try:
    # Model HF repo se download ho raha hy
    model_path = hf_hub_download(repo_id="aneela-pervez/My-Deepfake-Models", filename="Deepfake_Hybrid_SOTA.pth")
    state_dict = torch.load(model_path, map_location=device)
    
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print("✅ Image SOTA Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading image model weights: {e}")

# 4. Image Preprocessing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Prediction Function for Master Pipeline
def predict_image(img):
    if img is None:
        return {"Error": 1.0}
    
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        results = {
            "Fake (Deepfake)": float(probabilities[0]),
            "Real Image": float(probabilities[1])
        }
    return results