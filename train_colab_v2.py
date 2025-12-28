# GRAM Fine-tuning on Colab - CORRECT VERSION
# Upload this script and model.py from GRAM-main to Colab

"""
INSTRUCTIONS:
1. Upload these files to Colab:
   - model.py (from C:\fyp\GRAM-main\model.py)
   - MOE_epoch_2_v2.pth (your GRAM checkpoint)
   - data_combined_training.zip (your training data)

2. Run Cells in order
"""

# ==============================================================================
# CELL 1: Setup and Upload Files
# ==============================================================================
"""
# Mount Google Drive (optional - for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install timm tqdm

# Create working directories
!mkdir -p /content/fyp
%cd /content/fyp
"""

# ==============================================================================
# CELL 2: Upload files
# ==============================================================================
"""
from google.colab import files

print("Step 1: Upload model.py from GRAM-main folder")
uploaded = files.upload()

print("Step 2: Upload MOE_epoch_2_v2.pth checkpoint")
uploaded = files.upload()

print("Step 3: Upload data_combined_training.zip")
uploaded = files.upload()

# Unzip data
!unzip -q data_combined_training.zip -d /content/fyp/
"""

# ==============================================================================
# CELL 3: Import model (from uploaded model.py)
# ==============================================================================

import sys
sys.path.insert(0, '/content/fyp')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import the actual GRAM model
from model import mit_b5_MOE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ==============================================================================
# CELL 4: Dataset
# ==============================================================================

class SlumDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.img_dir.glob('*.png')))
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        
        img = Image.open(img_path).convert('RGB')
        
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', img.size, 0)
        
        if self.transform:
            img = self.transform(img)
            mask = T.ToTensor()(mask.resize((256, 256)))  # GRAM outputs 256x256
            mask = (mask > 0.5).float()
        
        return img, mask

# ==============================================================================
# CELL 5: Load GRAM Model with Checkpoint
# ==============================================================================

def load_gram_model(checkpoint_path):
    """Load GRAM model exactly like gram_loader.py does"""
    print(f"Loading GRAM model from {checkpoint_path}...")
    
    # Create model
    model = mit_b5_MOE(num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # Clean keys (remove 'module.' prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    # Load with strict=False to handle any mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    if len(missing) > 50:
        print("WARNING: Many missing keys - checkpoint may not match architecture!")
        print(f"First 10 missing: {missing[:10]}")
    
    return model

# ==============================================================================
# CELL 6: Training Functions
# ==============================================================================

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # GRAM model forward: returns (seg_out, dom_logits, MI_loss)
        # We use dummy domain labels (all zeros)
        batch_size = imgs.size(0)
        pseudo_domain = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        seg_out, dom_logits, mi_loss = model(imgs, pseudo_domain)
        
        # seg_out is (B, 2, 256, 256) - get class 1 probability
        probs = F.softmax(seg_out, dim=1)[:, 1:2]  # (B, 1, 256, 256)
        
        # Compute loss
        bce = F.binary_cross_entropy(probs, masks)
        dice = dice_loss(probs, masks)
        loss = bce + dice
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_iou = 0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            batch_size = imgs.size(0)
            pseudo_domain = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            seg_out, _, _ = model(imgs, pseudo_domain)
            probs = F.softmax(seg_out, dim=1)[:, 1:2]
            preds = (probs > 0.5).float()
            
            # IoU
            intersection = (preds * masks).sum()
            union = (preds + masks).clamp(0, 1).sum()
            iou = (intersection / (union + 1e-6)).item()
            total_iou += iou
    
    return total_iou / len(loader)

# ==============================================================================
# CELL 7: Main Training Loop  
# ==============================================================================

def main():
    # Config
    DATA_DIR = Path('/content/fyp/data_combined_training')
    CHECKPOINT_PATH = '/content/fyp/MOE_epoch_2_v2.pth'
    EPOCHS = 10
    BATCH_SIZE = 4
    LR = 1e-4
    
    print(f"Device: {DEVICE}")
    
    # Transforms (GRAM uses 512x512 input)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_ds = SlumDataset(DATA_DIR / 'train' / 'images', DATA_DIR / 'train' / 'masks', transform)
    val_ds = SlumDataset(DATA_DIR / 'val' / 'images', DATA_DIR / 'val' / 'masks', transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Load model with checkpoint
    model = load_gram_model(CHECKPOINT_PATH)
    model = model.to(DEVICE)
    
    # Optimizer - use lower LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_iou = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_iou = validate(model, val_loader, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),  # Match GRAM format
                'val_iou': val_iou
            }, '/content/fyp/best_gram_extended.pth')
            print(f"Saved best model (IoU: {val_iou:.4f})")
    
    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print("Download: /content/fyp/best_gram_extended.pth")

if __name__ == "__main__":
    main()

# ==============================================================================
# CELL 8: Download trained model
# ==============================================================================
"""
from google.colab import files
files.download('/content/fyp/best_gram_extended.pth')
"""
