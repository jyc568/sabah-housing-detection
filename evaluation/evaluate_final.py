"""
Final Evaluation with ROC Curves and Visualizations
Generates publication-quality figures for FYP report
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.insert(0, str(Path(r'C:\fyp\GRAM-main')))
from gram_loader import get_gram_model, gram_predict

# ---- CONFIG ----
DATA_ROOT = Path(r'C:\fyp\data_combined_training')
NEW_MODEL_CKPT = Path(r'C:\fyp\best_gram_extended.pth')
ORIGINAL_CKPT = Path(r'C:\fyp\GRAM-main\checkpoint\MOE_epoch_2_v2.pth')
FINETUNED_CKPT = Path(r'C:\fyp\finetune_gram_ckpt\best_gram_finetuned.pth')
OUTPUT_DIR = Path(r'C:\fyp\evaluation_results')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
THRESHOLD = 0.1

OUTPUT_DIR.mkdir(exist_ok=True)

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def has_annotations(mask_path, min_pixels=100):
    if not mask_path.exists():
        return False
    m = np.array(Image.open(mask_path).convert('L'))
    return (m > 127).sum() >= min_pixels

def compute_metrics(pred, gt, threshold=0.5):
    pred_binary = pred > threshold
    gt_binary = gt > 0
    
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    tn = np.logical_and(~pred_binary, ~gt_binary).sum()
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    return {
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }

def load_new_model(ckpt_path):
    from model import mit_b5_MOE
    print(f"Loading new model from {ckpt_path}...")
    model = mit_b5_MOE(num_classes=2)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def predict_new_model(model, inp_tensor):
    B = inp_tensor.size(0)
    pseudo_domain = torch.zeros(B, dtype=torch.long, device=inp_tensor.device)
    with torch.no_grad():
        seg_out, _, _ = model(inp_tensor, pseudo_domain)
        if seg_out.shape[2:] != (IMG_SIZE, IMG_SIZE):
            seg_out = F.interpolate(seg_out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        probs = F.softmax(seg_out, dim=1)[:, 1]
    return probs

def main():
    print("="*70)
    print("FINAL EVALUATION WITH ROC CURVES AND VISUALIZATIONS")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load models
    print("\nLoading models...")
    model_orig = get_gram_model(str(ORIGINAL_CKPT))
    model_ft = get_gram_model(str(FINETUNED_CKPT))
    model_new = load_new_model(str(NEW_MODEL_CKPT))
    print("All models loaded!\n")
    
    # Get test images
    test_img_dir = DATA_ROOT / 'test' / 'images'
    test_mask_dir = DATA_ROOT / 'test' / 'masks'
    test_images = sorted(test_img_dir.glob('*.png'))
    
    positive_tiles = [p for p in test_images if has_annotations(test_mask_dir / p.name)]
    negative_tiles = [p for p in test_images if not has_annotations(test_mask_dir / p.name)]
    
    print(f"Test images: {len(test_images)}")
    print(f"  Positive (slum): {len(positive_tiles)}")
    print(f"  Negative (formal): {len(negative_tiles)}")
    
    models = {
        'Original GRAM': (model_orig, gram_predict),
        'Fine-tuned': (model_ft, gram_predict),
        'Extended': (model_new, predict_new_model)
    }
    
    # Collect predictions for ROC curves
    all_probs = {name: [] for name in models}
    all_gts = []
    all_metrics = {name: [] for name in models}
    fpr_results = {name: [] for name in models}
    
    # Evaluate positive tiles
    print("\n" + "="*70)
    print("EVALUATING POSITIVE TILES")
    print("="*70)
    
    for img_path in tqdm(positive_tiles, desc="Positive tiles"):
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(DEVICE)
        
        mask_path = test_mask_dir / img_path.name
        gt = np.array(Image.open(mask_path).convert('L').resize((IMG_SIZE, IMG_SIZE)))
        gt = (gt > 127).astype(np.float32)
        all_gts.append(gt.flatten())
        
        for name, (model, predict_fn) in models.items():
            with torch.no_grad():
                prob = predict_fn(model, inp).squeeze().cpu().numpy()
            all_probs[name].append(prob.flatten())
            metrics = compute_metrics(prob, gt, THRESHOLD)
            all_metrics[name].append(metrics)
    
    # Evaluate negative tiles (for FPR)
    print("\nEvaluating negative tiles...")
    for img_path in tqdm(negative_tiles, desc="Negative tiles"):
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(DEVICE)
        
        for name, (model, predict_fn) in models.items():
            with torch.no_grad():
                prob = predict_fn(model, inp).squeeze().cpu().numpy()
            fp_rate = (prob >= THRESHOLD).sum() / prob.size * 100
            fpr_results[name].append(fp_rate)
    
    # ====== COMPUTE AND PRINT METRICS ======
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Positive Tiles)")
    print("="*70)
    
    results_table = []
    for name in models:
        metrics = all_metrics[name]
        avg = {k: np.mean([m[k] for m in metrics]) for k in ['iou', 'precision', 'recall', 'f1', 'accuracy', 'specificity']}
        avg_fpr = np.mean(fpr_results[name])
        
        results_table.append({
            'Model': name,
            'IoU': avg['iou'],
            'Precision': avg['precision'],
            'Recall': avg['recall'],
            'F1': avg['f1'],
            'Accuracy': avg['accuracy'],
            'FPR_Formal': avg_fpr
        })
        
        print(f"\n{name}:")
        print(f"  IoU:         {avg['iou']:.4f}")
        print(f"  Precision:   {avg['precision']:.4f}")
        print(f"  Recall:      {avg['recall']:.4f}")
        print(f"  F1 Score:    {avg['f1']:.4f}")
        print(f"  Accuracy:    {avg['accuracy']:.4f}")
        print(f"  FPR (Formal): {avg_fpr:.2f}%")
    
    # ====== GENERATE ROC CURVES ======
    print("\n" + "="*70)
    print("GENERATING ROC CURVES")
    print("="*70)
    
    plt.figure(figsize=(10, 8))
    colors = {'Original GRAM': 'blue', 'Fine-tuned': 'green', 'Extended': 'red'}
    
    y_true = np.concatenate(all_gts)
    
    for name in models:
        y_scores = np.concatenate(all_probs[name])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[name], lw=2, 
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Informal Settlement Detection', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150)
    plt.close()
    print(f"ROC curves saved to: {OUTPUT_DIR / 'roc_curves.png'}")
    
    # ====== GENERATE PRECISION-RECALL CURVES ======
    plt.figure(figsize=(10, 8))
    
    for name in models:
        y_scores = np.concatenate(all_probs[name])
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color=colors[name], lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Informal Settlement Detection', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'precision_recall_curves.png', dpi=150)
    plt.close()
    print(f"PR curves saved to: {OUTPUT_DIR / 'precision_recall_curves.png'}")
    
    # ====== GENERATE SAMPLE VISUALIZATIONS ======
    print("\nGenerating sample visualizations...")
    
    # Pick 4 sample positive tiles with good annotations
    sample_tiles = positive_tiles[:4]
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    for row, img_path in enumerate(sample_tiles):
        img = Image.open(img_path).convert('RGB')
        inp = transform(img).unsqueeze(0).to(DEVICE)
        
        mask_path = test_mask_dir / img_path.name
        gt = np.array(Image.open(mask_path).convert('L').resize((IMG_SIZE, IMG_SIZE)))
        gt = (gt > 127).astype(np.float32)
        
        # Column 0: Original image
        axes[row, 0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
        axes[row, 0].set_title('Input Image' if row == 0 else '')
        axes[row, 0].axis('off')
        
        # Column 1: Ground truth
        axes[row, 1].imshow(gt, cmap='Reds')
        axes[row, 1].set_title('Ground Truth' if row == 0 else '')
        axes[row, 1].axis('off')
        
        # Columns 2-4: Model predictions
        for col, (name, (model, predict_fn)) in enumerate(models.items(), start=2):
            with torch.no_grad():
                prob = predict_fn(model, inp).squeeze().cpu().numpy()
            pred = (prob >= THRESHOLD).astype(np.float32)
            
            axes[row, col].imshow(pred, cmap='Reds')
            axes[row, col].set_title(name if row == 0 else '')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_predictions.png', dpi=150)
    plt.close()
    print(f"Sample predictions saved to: {OUTPUT_DIR / 'sample_predictions.png'}")
    
    # ====== SAVE NUMERIC RESULTS TO FILE ======
    with open(OUTPUT_DIR / 'evaluation_metrics.txt', 'w') as f:
        f.write("INFORMAL SETTLEMENT DETECTION - EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Set: {len(positive_tiles)} positive, {len(negative_tiles)} negative tiles\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        
        f.write("POSITIVE TILES (Informal Settlements)\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Model':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}\n")
        f.write("-"*60 + "\n")
        for r in results_table:
            f.write(f"{r['Model']:<20} {r['IoU']:>8.4f} {r['Precision']:>8.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}\n")
        
        f.write("\n\nNEGATIVE TILES (Formal Housing) - False Positive Rate\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Model':<20} {'Avg FPR':>10} {'Max FPR':>10}\n")
        f.write("-"*60 + "\n")
        for name in models:
            avg_fpr = np.mean(fpr_results[name])
            max_fpr = np.max(fpr_results[name])
            f.write(f"{name:<20} {avg_fpr:>9.2f}% {max_fpr:>9.2f}%\n")
    
    print(f"\nMetrics saved to: {OUTPUT_DIR / 'evaluation_metrics.txt'}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nOutput files in: {OUTPUT_DIR}")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")  
    print("  - sample_predictions.png")
    print("  - evaluation_metrics.txt")

if __name__ == "__main__":
    main()
