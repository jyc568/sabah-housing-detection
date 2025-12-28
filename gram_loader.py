import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add GRAM-main to python path to import its modules
GRAM_PATH = Path(r'C:\fyp\GRAM-main')
if str(GRAM_PATH) not in sys.path:
    sys.path.append(str(GRAM_PATH))

# Import model definition
try:
    from model import mit_b5_MOE
except ImportError as e:
    print(f"Error importing GRAM model: {e}")
    # Fallback/Debug
    raise

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gram_model(ckpt_path):
    """
    Loads the GRAM MoE model (mit_b5_MOE).
    """
    print(f"Loading GRAM model from {ckpt_path}...")
    
    # Instantiate model
    # We assume 'num_classes=2' (Binary: Background vs Informal)
    # We assume 'num_domains=3' or '12'? 
    # Use strict=False to ignore domain classifier head mismatch if specific count differs.
    model = mit_b5_MOE(num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
        
    # Clean keys if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
        
    # Load
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"GRAM Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    # If crucial encoder parts are missing, we have a problem.
    # mit_b5 encoder keys start with 'block1', 'patch_embed1', etc.
    
    model.to(DEVICE)
    model.eval()
    return model

def gram_predict(model, input_tensor):
    """
    input_tensor: (B, 3, H, W) normalized
    Returns: (B, H, W) probability map (0-1) for class 1
    """
    model.eval()
    B, C, H, W = input_tensor.shape
    
    # Dummy domain label: 0
    pseudo_domain_label = torch.zeros(B, dtype=torch.long, device=input_tensor.device)
    
    with torch.no_grad():
        # Forward pass
        # returns: seg_out, dom_logits, total_MI
        seg_out, _, _ = model(input_tensor, pseudo_domain_label)
        
        # seg_out shape: (B, 2, 256, 256) (Internal hardcoded 256)
        
        # Upsample to input size (512x512)
        if seg_out.shape[2:] != (H, W):
            seg_out = F.interpolate(seg_out, size=(H, W), mode='bilinear', align_corners=False)
            
        # Softmax to get probability
        probs = F.softmax(seg_out, dim=1) # (B, 2, H, W)
        
        # Return class 1 probability
        prob_map = probs[:, 1, :, :] # (B, H, W)
        
    return prob_map
