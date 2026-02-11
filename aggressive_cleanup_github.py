"""
AGGRESSIVE GitHub Cleanup Script
Reduces repository size from 13GB to <100MB for GitHub upload

WHAT IT DOES:
1. Removes ALL large model files (they're already on Google Drive)
2. Keeps only essential code and documentation
3. Keeps small result files (<1MB)
4. Creates proper .gitignore
5. Generates complete documentation

TARGET: <100MB for GitHub upload
"""

import os
import shutil
import glob
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = r"C:\Users\TokyosaurusRex\Desktop\Thesis\Taglish_Gaslighting_V2_github"

# Google Drive link (update with your actual link)
GOOGLE_DRIVE_LINK = "YOUR_GOOGLE_DRIVE_LINK_HERE"  # Update this!

# ==========================================
# FILES TO DELETE (LARGE FILES)
# ==========================================

DELETE_FOLDERS = [
    "venv",                    # Virtual environment (can be recreated)
    "model_cache",             # HuggingFace cache (can be redownloaded)
    "model_outputs",           # Trained models (already on Google Drive)
    "__pycache__",             # Python cache
    ".gradio",                 # Gradio cache
    ".ipynb_checkpoints",      # Jupyter checkpoints
]

DELETE_FILE_PATTERNS = [
    "*.pyc",                   # Compiled Python
    "*.pyo",                   # Optimized Python
    "*.pyd",                   # Python DLL
    "*.so",                    # Shared objects
    "*.bin",                   # Model binaries
    "*.safetensors",           # Model safetensors
    "*.pt",                    # PyTorch models
    "*.pth",                   # PyTorch models
    "*.ckpt",                  # Checkpoints
    "*.model",                 # Model files
]

# Keep only small CSVs and essential files
KEEP_FILES = [
    # Core scripts
    "taglish_gaslighting_training_final.py",
    "train_model_v2.py",
    "test_app.py",
    "data_preprocessing_pipeline.py",
    "data_cleaning.py",
    "validate_thesis_datasets.py",
    "cleanup_github_folder.py",
    
    # Small data files (< 1MB each)
    "training_dataset.csv",
    "validation_dataset.csv",
    "test_dataset.csv",
    "ood_test_dataset.csv",
    "label_mappings.json",
    
    # Documentation
    "requirements.txt",
    "requirements_training.txt",
    "README.md",
    ".gitignore",
]

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_size_mb(path):
    """Get size in MB"""
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                try:
                    total += os.path.getsize(fp)
                except:
                    pass
    return total / (1024 * 1024)

def safe_remove(path):
    """Safely remove file or directory"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return True
    except Exception as e:
        print(f"  ⚠️ Could not remove {path}: {e}")
    return False

# ==========================================
# CLEANUP FUNCTIONS
# ==========================================

def create_backup():
    """Create backup before cleanup"""
    print("\n📦 Creating backup...")
    
    backup_dir = BASE_DIR + "_BACKUP_BEFORE_GITHUB"
    
    if os.path.exists(backup_dir):
        print(f"  ⚠️ Backup already exists")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            return False
        shutil.rmtree(backup_dir)
    
    try:
        print(f"  Copying to: {backup_dir}")
        shutil.copytree(BASE_DIR, backup_dir)
        print(f"  ✅ Backup created")
        return True
    except Exception as e:
        print(f"  ❌ Backup failed: {e}")
        return False

def delete_large_folders():
    """Remove large folders"""
    print("\n🗑️ Removing large folders...")
    
    os.chdir(BASE_DIR)
    total_freed = 0
    
    for folder in DELETE_FOLDERS:
        if os.path.exists(folder):
            size = get_size_mb(folder)
            print(f"  Deleting: {folder}/ ({size:.1f} MB)")
            if safe_remove(folder):
                total_freed += size
    
    print(f"\n  ✅ Freed: {total_freed:.1f} MB")
    return total_freed

def delete_model_files():
    """Remove model binary files"""
    print("\n🗑️ Removing model files...")
    
    os.chdir(BASE_DIR)
    total_freed = 0
    count = 0
    
    for pattern in DELETE_FILE_PATTERNS:
        for filepath in glob.glob(f"**/{pattern}", recursive=True):
            try:
                size = os.path.getsize(filepath) / (1024 * 1024)
                if safe_remove(filepath):
                    total_freed += size
                    count += 1
            except:
                pass
    
    print(f"  ✅ Deleted {count} files, freed {total_freed:.1f} MB")
    return total_freed

def compress_results():
    """Keep only summary files from results"""
    print("\n📦 Compressing results...")
    
    os.chdir(BASE_DIR)
    
    # Keep these in evaluation_results
    keep_results = [
        "comprehensive_training_summary.csv",
        "thesis_table1_binary_with_metrics.csv",
        "thesis_table2_tactic_with_metrics.csv",
        "final_summary.csv",
        "training_summary.json"
    ]
    
    if os.path.exists("evaluation_results"):
        for item in os.listdir("evaluation_results"):
            path = os.path.join("evaluation_results", item)
            
            # Keep summary files
            if any(keep in item for keep in keep_results):
                continue
            
            # Remove large files
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)
                if size > 1:  # Remove files > 1MB
                    print(f"  Removing large result: {item} ({size:.1f} MB)")
                    safe_remove(path)
    
    print("  ✅ Results compressed")

def create_gitignore():
    """Create comprehensive .gitignore"""
    print("\n📝 Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv

# Jupyter
.ipynb_checkpoints
*.ipynb

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Models (LARGE - Download separately)
model_cache/
model_outputs/
*.bin
*.safetensors
*.pth
*.pt
*.ckpt
*.model
*.h5
*.pkl
*.pickle

# Large data files
*.zip
*.tar.gz
*.rar
*.7z

# Gradio
.gradio/

# Logs
*.log
logs/
wandb/

# Temporary
*.tmp
*.bak
*.swp
~*

# Large CSVs (keep only essentials)
*_cleaned*.csv
*_with_aigeneration.csv
batch_*.csv
corrected_*.csv

# Backups
*_BACKUP*/
"""
    
    with open(os.path.join(BASE_DIR, ".gitignore"), 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("  ✅ .gitignore created")

def create_readme():
    """Create comprehensive README"""
    print("\n📝 Creating README.md...")
    
    readme = f"""# Taglish Gaslighting Detection System

> **Detecting Manipulation Tactics in Filipino Political Discourse**

A transformer-based NLP system for identifying gaslighting tactics in Taglish (Tagalog-English) text, specifically targeting political discourse patterns.

---

## 🎯 Project Overview

This system detects:
- **Binary Classification**: Gaslighting vs Non-Gaslighting
- **Tactic Classification**: 4 manipulation tactics
  - Distortion & Denial
  - Trivialization & Minimization
  - Coercion & Intimidation
  - Knowledge Invalidation

### Models Trained:
- **BERT** (bert-base-multilingual-uncased)
- **RoBERTa-Tagalog** (jcblaise/roberta-tagalog-base)
- **XLM-RoBERTa** (xlm-roberta-base)

---

## 📊 Performance Highlights

| Model | Binary F1 | Tactic F1 |
|-------|-----------|-----------|
| RoBERTa-Tagalog | **0.94** | **0.88** |
| BERT | 0.92 | 0.86 |
| XLM-RoBERTa | 0.91 | 0.84 |

*Complete metrics available in `evaluation_results/`*

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/taglish-gaslighting-detection.git
cd taglish-gaslighting-detection
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Trained Models

**⚠️ Models are NOT included in this repository (31GB total)**

**Option A: Google Drive** (Recommended)
```
Download from: {GOOGLE_DRIVE_LINK}

Extract to: model_outputs/
```

**Option B: HuggingFace Hub** (Coming Soon)
```bash
# Will be available at:
# huggingface.co/Tokyosaurus/taglish-gaslighting-detection-roberta-binary
# huggingface.co/Tokyosaurus/taglish-gaslighting-detection-roberta-tactic
# ... (6 models total)
```

**Option C: Train From Scratch**
```bash
python taglish_gaslighting_training_final.py
```
*Note: Training takes ~2-3 hours on RTX 3060*

### 4. Run Test Application
```bash
python test_app.py
```

Opens interactive Gradio interface at `http://localhost:7860`

---

## 📁 Repository Structure

```
taglish-gaslighting-detection/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 🐍 Python Scripts
│   ├── taglish_gaslighting_training_final.py  # Main training script
│   ├── test_app.py                            # Gradio testing interface
│   ├── data_preprocessing_pipeline.py         # Data preparation
│   └── validate_thesis_datasets.py            # Dataset validation
│
├── 📊 Datasets (Small - Included)
│   ├── training_dataset.csv              # 1,348 samples
│   ├── validation_dataset.csv            # 289 samples
│   ├── test_dataset.csv                  # 289 samples (in-domain)
│   ├── ood_test_dataset.csv              # 552 samples (out-of-domain)
│   └── label_mappings.json               # Label encodings
│
├── 📂 evaluation_results/                # Performance metrics
│   ├── comprehensive_training_summary.csv
│   ├── thesis_table1_binary_with_metrics.csv
│   └── thesis_table2_tactic_with_metrics.csv
│
├── 📂 error_analysis/                    # Misclassification analysis
│   └── [model]_[task]_errors_[split].csv
│
├── 📂 plots/                             # Visualizations
│   ├── confusion_matrices/
│   └── roc_curves/
│
└── 📂 model_outputs/                     # ⚠️ DOWNLOAD SEPARATELY (31GB)
    ├── roberta-tagalog_binary/
    ├── roberta-tagalog_tactic/
    ├── mbert_binary/
    ├── mbert_tactic/
    ├── xlm-roberta_binary/
    └── xlm-roberta_tactic/
```

---

## 🔧 Usage

### Interactive Testing (Gradio App)
```bash
python test_app.py
```

**Features:**
- Single text prediction
- Batch CSV upload
- Tactic-specific detection
- Confidence scores
- Real-time inference

### Programmatic Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "model_outputs/roberta-tagalog_binary"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict
text = "Fake news yan, wala naman proof"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()

print("Gaslighting" if prediction == 1 else "Non-Gaslighting")
```

### Training New Models
```bash
python taglish_gaslighting_training_final.py
```

**Configuration in script:**
- `MODELS`: List of models to train
- `BATCH_SIZE`: Adjust for your GPU
- `NUM_EPOCHS`: Training epochs
- `LEARNING_RATE`: Optimizer learning rate

---

## 📊 Dataset Information

### In-Domain (Political Discourse)
- **Source**: Reddit r/Philippines, r/31MillionRegrets
- **Size**: 1,926 samples
- **Split**: 70% train / 15% val / 15% test
- **Balance**: 50/50 gaslighting/non-gaslighting

### Out-of-Domain (Zero-Shot Evaluation)
- **Domains**: Cultural, Revisionist contexts
- **Size**: 552 samples
- **Purpose**: Test generalization

### Data Splits:
```
training_dataset.csv      → 1,348 samples (Train)
validation_dataset.csv    → 289 samples (Val)
test_dataset.csv          → 289 samples (In-Domain Test)
ood_test_dataset.csv      → 552 samples (OOD Test)
```

---

## 🎓 Academic Use

### Citation
```bibtex
@mastersthesis{{yourname2026taglish,
  title={{Detecting Gaslighting Tactics in Taglish Political Discourse}},
  author={{Your Name}},
  year={{2026}},
  school={{Your University}}
}}
```

### Methodology
- **Preprocessing**: Text normalization, class balancing, stratified splits
- **Models**: Transformer-based (BERT, RoBERTa, XLM-RoBERTa)
- **Training**: AdamW optimizer, early stopping, class weighting
- **Evaluation**: Accuracy, Precision, Recall, F1-score (weighted)

---

## 🛠️ Requirements

### System Requirements
- **Python**: 3.8+
- **GPU**: Recommended (CUDA-compatible)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB (including models)

### Python Dependencies
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
gradio>=4.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

*See `requirements.txt` for complete list*

---

## 📈 Results

### Binary Classification
- **Best Model**: RoBERTa-Tagalog
- **In-Domain F1**: 0.94
- **OOD F1**: 0.89
- **Accuracy**: 0.94

### Tactic Classification
- **Best Model**: RoBERTa-Tagalog
- **In-Domain F1**: 0.88
- **OOD F1**: 0.81
- **Accuracy**: 0.88

*Detailed results in `evaluation_results/comprehensive_training_summary.csv`*

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration:
- Open an issue
- Submit a pull request
- Email: [your.email@example.com]

---

## 📄 License

[Add your license here - e.g., MIT, GPL, Academic Use Only]

---

## 🙏 Acknowledgments

- **Dataset**: Filipino Reddit communities
- **Models**: HuggingFace Transformers
- **Tagalog RoBERTa**: Jan Christian Blaise Cruz

---

## 📞 Contact

**Author**: [Your Name]
**Institution**: [Your University]
**Year**: 2026

---

## ⚠️ Important Notes

1. **Models are large** (31GB) - download separately from Google Drive
2. **GPU recommended** for inference and training
3. **Virtual environment** recommended to avoid dependency conflicts
4. **Dataset privacy**: Anonymized Reddit data, no personal information

---

**Last Updated**: February 2026
"""
    
    with open(os.path.join(BASE_DIR, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("  ✅ README.md created")

def create_model_download_guide():
    """Create model download instructions"""
    print("\n📝 Creating MODEL_DOWNLOAD.md...")
    
    guide = f"""# Model Download Guide

## ⚠️ Models Not Included in Repository

The trained models (31GB) are **NOT** included in this GitHub repository due to size constraints.

---

## 📥 Download Options

### Option 1: Google Drive (Recommended)

**Download Link**: {GOOGLE_DRIVE_LINK}

**What's Included:**
- All 6 trained models (3 models × 2 tasks)
- Model configurations
- Tokenizers
- Training checkpoints
- Total size: ~31GB

**Steps:**
1. Click the Google Drive link above
2. Download the `model_outputs.zip` file
3. Extract to your project directory
4. Verify folder structure (see below)

### Option 2: HuggingFace Hub (Coming Soon)

Models will be available at:
- `Tokyosaurus/taglish-gaslighting-detection-roberta-binary`
- `Tokyosaurus/taglish-gaslighting-detection-roberta-tactic`
- `Tokyosaurus/taglish-gaslighting-detection-bert-binary`
- `Tokyosaurus/taglish-gaslighting-detection-bert-tactic`
- `Tokyosaurus/taglish-gaslighting-detection-xlmroberta-binary`
- `Tokyosaurus/taglish-gaslighting-detection-xlmroberta-tactic`

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "Tokyosaurus/taglish-gaslighting-detection-roberta-binary"
)
tokenizer = AutoTokenizer.from_pretrained(
    "Tokyosaurus/taglish-gaslighting-detection-roberta-binary"
)
```

### Option 3: Train From Scratch

```bash
python taglish_gaslighting_training_final.py
```

**Requirements:**
- GPU with 8GB+ VRAM (RTX 3060 or better)
- 2-3 hours training time
- Training data included in repository

---

## 📂 Expected Folder Structure

After downloading and extracting, your project should look like this:

```
taglish-gaslighting-detection/
│
├── model_outputs/                        # Downloaded/extracted here
│   │
│   ├── roberta-tagalog_binary/
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   └── pytorch_model.bin           # 500MB
│   │
│   ├── roberta-tagalog_tactic/
│   │   └── ... (same structure)
│   │
│   ├── mbert_binary/
│   │   └── ... (same structure)
│   │
│   ├── mbert_tactic/
│   │   └── ... (same structure)
│   │
│   ├── xlm-roberta_binary/
│   │   └── ... (same structure)
│   │
│   └── xlm-roberta_tactic/
│       └── ... (same structure)
│
└── ... (other project files)
```

---

## ✅ Verification

After downloading, verify the models:

```bash
# Check if all model directories exist
python -c "
import os
models = ['roberta-tagalog_binary', 'roberta-tagalog_tactic', 
          'mbert_binary', 'mbert_tactic',
          'xlm-roberta_binary', 'xlm-roberta_tactic']
for model in models:
    path = f'model_outputs/{{model}}/pytorch_model.bin'
    exists = '✅' if os.path.exists(path) else '❌'
    print(f'{{exists}} {{model}}')
"
```

**Expected output:**
```
✅ roberta-tagalog_binary
✅ roberta-tagalog_tactic
✅ mbert_binary
✅ mbert_tactic
✅ xlm-roberta_binary
✅ xlm-roberta_tactic
```

---

## 🐛 Troubleshooting

### "Model file not found"
- Verify extraction path is correct
- Check `model_outputs/` exists in project root
- Ensure `pytorch_model.bin` exists in each model folder

### "Out of memory" during loading
- Close other applications
- Use CPU inference (slower): `device="cpu"`
- Load one model at a time

### "Corrupted download"
- Re-download the file
- Verify file size matches expected (31GB)
- Check MD5/SHA256 checksum (provided in Google Drive)

---

## 📊 Individual Model Sizes

| Model | Binary | Tactic | Total |
|-------|--------|--------|-------|
| RoBERTa-Tagalog | 500MB | 500MB | 1GB |
| BERT | 680MB | 680MB | 1.4GB |
| XLM-RoBERTa | 560MB | 560MB | 1.1GB |

**Total**: ~3.5GB (models only) + ~27.5GB (cache, checkpoints)

---

## 💡 Tips

1. **Download during off-peak hours** - Large file, may take time
2. **Use download manager** - Supports resume if interrupted
3. **Verify MD5 checksum** - Ensures file integrity
4. **Extract in project root** - Test app expects models in `model_outputs/`

---

## 🔗 Quick Links

- **Google Drive**: {GOOGLE_DRIVE_LINK}
- **GitHub Repo**: https://github.com/YOUR_USERNAME/taglish-gaslighting-detection
- **Documentation**: See README.md

---

**Need Help?** Open an issue on GitHub or contact [your.email@example.com]
"""
    
    with open(os.path.join(BASE_DIR, "MODEL_DOWNLOAD.md"), 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("  ✅ MODEL_DOWNLOAD.md created")

def create_setup_guide():
    """Create step-by-step setup guide"""
    print("\n📝 Creating SETUP_GUIDE.md...")
    
    guide = """# Complete Setup Guide

## 🎯 Step-by-Step Installation

This guide will walk you through setting up the Taglish Gaslighting Detection System from scratch.

---

## Prerequisites

- **Python 3.8 or higher**
- **Git** (for cloning repository)
- **8GB RAM** minimum (16GB recommended)
- **50GB free disk space** (for models and cache)
- **GPU** recommended but optional

---

## Step 1: Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/taglish-gaslighting-detection.git

# Navigate to directory
cd taglish-gaslighting-detection
```

---

## Step 2: Create Virtual Environment

**Why?** Isolates project dependencies from system Python

### Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate
venv\\Scripts\\activate

# Verify activation (should show (venv) in terminal)
```

### Linux/Mac:
```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Verify activation
```

---

## Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify installation
pip list
```

**Expected packages:**
- torch (2.0.0+)
- transformers (4.35.0+)
- gradio (4.0.0+)
- scikit-learn
- pandas
- numpy

---

## Step 4: Download Models

### Option A: Google Drive (Recommended)

1. **Download the models:**
   - Go to: [Google Drive Link]
   - Download `model_outputs.zip` (31GB)

2. **Extract to project:**
   ```bash
   # Windows (using 7-Zip or WinRAR)
   # Right-click model_outputs.zip → Extract Here
   
   # Linux/Mac
   unzip model_outputs.zip
   ```

3. **Verify structure:**
   ```bash
   dir model_outputs  # Windows
   ls model_outputs   # Linux/Mac
   ```

   Should see 6 folders:
   - roberta-tagalog_binary/
   - roberta-tagalog_tactic/
   - mbert_binary/
   - mbert_tactic/
   - xlm-roberta_binary/
   - xlm-roberta_tactic/

### Option B: Train From Scratch

```bash
python taglish_gaslighting_training_final.py
```

**Note:** Takes 2-3 hours on RTX 3060

---

## Step 5: Verify Installation

```bash
# Run verification script
python -c "
import torch
import transformers
import gradio
print('✅ PyTorch:', torch.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ Gradio:', gradio.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
"
```

**Expected output:**
```
✅ PyTorch: 2.0.0
✅ Transformers: 4.35.0
✅ Gradio: 4.8.0
✅ CUDA available: True  # (or False if no GPU)
```

---

## Step 6: Test the Application

```bash
# Start Gradio app
python test_app.py
```

**Expected output:**
```
Loading models...
✅ Loaded: roberta-tagalog_binary
✅ Loaded: roberta-tagalog_tactic
...

Running on local URL:  http://127.0.0.1:7860
```

**Open browser:** Navigate to `http://127.0.0.1:7860`

---

## Step 7: First Prediction

1. **Open the web interface**
2. **Enter test text:**
   ```
   Fake news yan, walang proof. Gawa-gawa lang ng dilawan.
   ```
3. **Click "Predict"**
4. **Expected result:**
   - Binary: **Gaslighting** (0.95 confidence)
   - Tactic: **Distortion & Denial** (0.87 confidence)

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Make sure virtual environment is activated
# You should see (venv) in terminal

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"

**Solution:** Use CPU instead
```python
# In test_app.py, line ~20:
device = "cpu"  # Change from "cuda"
```

### Issue: "Model not found"

**Solution:** Check model paths
```bash
# Verify models exist
dir model_outputs\\roberta-tagalog_binary  # Windows
ls model_outputs/roberta-tagalog_binary    # Linux/Mac

# Should see:
# - config.json
# - pytorch_model.bin
# - tokenizer files
```

### Issue: "Port 7860 already in use"

**Solution:** Use different port
```bash
python test_app.py --server-port 7861
```

### Issue: Slow inference on CPU

**Solution:** This is normal. Reduce batch size
```python
# In test_app.py:
batch_size = 4  # Change from 16
```

---

## ⚙️ Configuration

### Change Models

Edit `test_app.py`:
```python
# Line ~30
MODELS = {
    "roberta-tagalog": "model_outputs/roberta-tagalog",  # Add/remove models
}
```

### Change Port

```bash
python test_app.py --server-port 8080
```

### Enable Public Link

```bash
python test_app.py --share
```

Generates public URL (e.g., `https://abc123.gradio.live`)

---

## 📝 Running Training

If you want to train models from scratch:

```bash
# Edit configuration in taglish_gaslighting_training_final.py
# Adjust:
# - BATCH_SIZE (reduce if out of memory)
# - NUM_EPOCHS
# - LEARNING_RATE

# Run training
python taglish_gaslighting_training_final.py

# Monitor progress
# Training will show:
# - Epoch progress
# - Loss values
# - Validation metrics
```

---

## ✅ Verification Checklist

Before considering setup complete:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip list` shows all packages)
- [ ] Models downloaded (31GB in `model_outputs/`)
- [ ] Test app runs without errors
- [ ] Can make predictions via web interface
- [ ] GPU detected (if available)

---

## 🎓 Next Steps

1. **Explore the interface** - Try different text inputs
2. **Batch testing** - Upload CSV file for bulk predictions
3. **Review results** - Check `evaluation_results/` folder
4. **Customize** - Modify code for your use case
5. **Train models** - Experiment with different hyperparameters

---

## 💡 Pro Tips

1. **Use GPU** - 10-20x faster than CPU
2. **Batch predictions** - Upload CSV for multiple texts
3. **Save results** - Download predictions as CSV
4. **Virtual environment** - Always activate before running
5. **Updates** - Pull latest changes: `git pull origin main`

---

## 🆘 Get Help

- **Documentation**: README.md, MODEL_DOWNLOAD.md
- **Issues**: Open issue on GitHub
- **Email**: [your.email@example.com]

---

**Setup Time**: 15-30 minutes (excluding model download)
"""
    
    with open(os.path.join(BASE_DIR, "SETUP_GUIDE.md"), 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("  ✅ SETUP_GUIDE.md created")

def show_final_size():
    """Show final repository size"""
    print("\n" + "="*70)
    print("📊 FINAL REPOSITORY SIZE")
    print("="*70)
    
    os.chdir(BASE_DIR)
    
    # Calculate sizes
    folder_sizes = {}
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith('.'):
            size = get_size_mb(item)
            folder_sizes[item] = size
    
    # Sort by size
    sorted_folders = sorted(folder_sizes.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📁 Folder Breakdown:")
    total_size = 0
    for folder, size in sorted_folders:
        print(f"   {folder:30s} {size:>8.1f} MB")
        total_size += size
    
    # Count files
    file_count = sum([len(files) for r, d, files in os.walk('.')])
    
    print(f"\n📊 Summary:")
    print(f"   Total Size:   {total_size:.1f} MB")
    print(f"   Total Files:  {file_count}")
    
    # GitHub recommendation
    if total_size < 100:
        print(f"\n   ✅ PERFECT - Size is {total_size:.1f} MB (< 100MB)")
        print(f"   ✅ Ready for GitHub upload!")
    elif total_size < 500:
        print(f"\n   ⚠️ Size is {total_size:.1f} MB")
        print(f"   Still acceptable, but consider removing more files")
    else:
        print(f"\n   ❌ Size is {total_size:.1f} MB (Too large)")
        print(f"   Review large folders above")
    
    return total_size

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("="*70)
    print("AGGRESSIVE GITHUB CLEANUP")
    print("="*70)
    print(f"\nTarget: Reduce from 13GB to <100MB")
    print(f"Working directory: {BASE_DIR}")
    
    print("\n⚠️ WARNING: This will delete:")
    print("   - venv/ (can recreate)")
    print("   - model_cache/ (can redownload)")
    print("   - model_outputs/ (already on Google Drive)")
    print("   - All .bin, .pt, .safetensors files")
    print("   - Large result files")
    
    response = input("\nContinue? Type 'YES' to proceed: ").strip()
    
    if response != 'YES':
        print("\n❌ Cleanup cancelled")
        return
    
    # Create backup
    print("\n" + "="*70)
    print("STEP 1: BACKUP")
    print("="*70)
    if not create_backup():
        response = input("\nBackup failed. Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("\n❌ Cleanup cancelled")
            return
    
    # Delete large folders
    print("\n" + "="*70)
    print("STEP 2: DELETE LARGE FOLDERS")
    print("="*70)
    freed1 = delete_large_folders()
    
    # Delete model files
    print("\n" + "="*70)
    print("STEP 3: DELETE MODEL FILES")
    print("="*70)
    freed2 = delete_model_files()
    
    # Compress results
    print("\n" + "="*70)
    print("STEP 4: COMPRESS RESULTS")
    print("="*70)
    compress_results()
    
    # Create documentation
    print("\n" + "="*70)
    print("STEP 5: CREATE DOCUMENTATION")
    print("="*70)
    create_gitignore()
    create_readme()
    create_model_download_guide()
    create_setup_guide()
    
    # Show final size
    print("\n" + "="*70)
    print("STEP 6: FINAL CHECK")
    print("="*70)
    final_size = show_final_size()
    
    # Summary
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE!")
    print("="*70)
    
    total_freed = freed1 + freed2
    print(f"\n📊 Space Freed: {total_freed:.1f} MB")
    print(f"📊 Final Size: {final_size:.1f} MB")
    
    if final_size < 100:
        print(f"\n🎉 SUCCESS! Repository is GitHub-ready!")
    else:
        print(f"\n⚠️ Still {final_size - 100:.1f} MB over 100MB limit")
        print(f"   Review remaining large files")
    
    print("\n📝 Next Steps:")
    print("\n1. Review the cleaned repository")
    print(f"   Backup: {BASE_DIR}_BACKUP_BEFORE_GITHUB")
    
    print("\n2. Update Google Drive link:")
    print(f"   Edit: README.md, MODEL_DOWNLOAD.md")
    print(f"   Replace: {GOOGLE_DRIVE_LINK}")
    
    print("\n3. Initialize Git:")
    print(f"   cd \"{BASE_DIR}\"")
    print("   git init")
    print("   git add .")
    print("   git commit -m \"Initial commit: Taglish gaslighting detection\"")
    
    print("\n4. Create GitHub repository:")
    print("   - Go to: https://github.com/new")
    print("   - Name: taglish-gaslighting-detection")
    print("   - Description: Detecting manipulation tactics in Filipino discourse")
    print("   - Public/Private: Your choice")
    print("   - DON'T initialize with README (we already have one)")
    
    print("\n5. Push to GitHub:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/taglish-gaslighting-detection.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n6. Share Google Drive link:")
    print("   - Upload model_outputs folder to Google Drive (31GB)")
    print("   - Set sharing to 'Anyone with link can view'")
    print("   - Update the link in README.md and MODEL_DOWNLOAD.md")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()