# Complete Setup Guide

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
git clone https://github.com/Jtugado/taglish-gaslighting-detection.git

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
venv\Scripts\activate

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
   - Go to: https://drive.google.com/drive/folders/1TEbQ7TyzqI7wmLDuighQ4CRUO0S-zw6C?usp=sharing
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
dir model_outputs\roberta-tagalog_binary  # Windows
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
