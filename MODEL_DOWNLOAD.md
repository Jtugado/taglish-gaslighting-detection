# Model Download Guide

## ⚠️ Models Not Included in Repository

The trained models (31GB) are **NOT** included in this GitHub repository due to size constraints.

---

## 📥 Download Options

### Option 1: Google Drive (Recommended)

**Download Link**: https://drive.google.com/drive/folders/1TEbQ7TyzqI7wmLDuighQ4CRUO0S-zw6C?usp=drive_link

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
    path = f'model_outputs/{model}/pytorch_model.bin'
    exists = '✅' if os.path.exists(path) else '❌'
    print(f'{exists} {model}')
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

- **Google Drive**: https://drive.google.com/drive/folders/1TEbQ7TyzqI7wmLDuighQ4CRUO0S-zw6C?usp=drive_link
- **GitHub Repo**: https://github.com/Jtugado/taglish-gaslighting-detection
- **Documentation**: See README.md

---

**Need Help?** Open an issue on GitHub or contact [your.email@example.com]
