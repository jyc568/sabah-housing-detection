# Sabah Informal Housing Detection System

AI-powered dashboard for detecting informal settlements in Sabah, Malaysia using satellite imagery and deep learning.

## 🎯 Features

- **Interactive Map Dashboard** - Click anywhere to analyze 512×512 areas
- **GRAM Ensemble Model** - Combines original and extended models
- **Cloud Database** - Tile index stored in Supabase (PostGIS)
- **Live Tile Fallback** - Fetches Sentinel-2 imagery when needed

## 📁 Project Structure

```
sabah-housing-detection/
├── snapshot_server.py       # Main Flask server
├── gram_loader.py           # Model loading utilities
├── tile_index.geojson       # Tile spatial index
├── static/
│   └── dashboard.html       # Web interface
├── GRAM-main/
│   └── model.py             # Neural network architecture
├── checkpoints/             # Model files (download separately)
│   ├── MOE_epoch_2_v2.pth
│   └── best_gram_extended.pth
├── evaluation/
│   ├── evaluate_final.py    # Evaluation script
│   └── results/             # ROC/PR curves, metrics
└── docs/
    └── chapter4_diagrams.md # UML diagrams
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/jyc568/sabah-housing-detection.git
cd sabah-housing-detection
pip install -r requirements.txt
```

### 2. Download Model Checkpoints
Download from [GitHub Releases](../../releases) and place in `checkpoints/`:
- `MOE_epoch_2_v2.pth` (~98 MB)
- `best_gram_extended.pth` (~98 MB)

### 3. Verify Setup
```bash
python setup_check.py
```

### 4. Run Server
```bash
python snapshot_server.py
```

### 5. Open Dashboard
Navigate to `http://127.0.0.1:5000`

## 📊 Model Performance

| Model | IoU | Precision | Recall | F1 | FPR |
|-------|-----|-----------|--------|-----|-----|
| Original GRAM | 0.65 | 0.65 | 0.98 | 0.75 | 32.3% |
| Extended GRAM | 0.71 | 0.75 | 0.90 | 0.79 | 13.0% |
| **Ensemble** | **0.73** | **0.76** | **0.91** | **0.81** | **12.5%** |

## ⚙️ Configuration

Edit `snapshot_server.py`:
```python
USE_CLOUD_DB = True      # Use Supabase cloud
WEIGHT_ORIGINAL = 0.2    # Original model weight
WEIGHT_EXTENDED = 0.8    # Extended model weight
THRESH = 0.1             # Detection threshold
```

## 📝 License

MIT License - Academic use only.

## 🙏 Acknowledgments

- GRAM model from [SiswiMon](https://github.com/SiswiMon/GRAM)
- Sentinel-2 imagery from EOX/Copernicus
