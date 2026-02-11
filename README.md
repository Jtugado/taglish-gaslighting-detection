# Taglish Gaslighting Detection System

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
- **RoBERTa-Tagalog** (`jcblaise/roberta-tagalog-base`) - *Best Performer*
- **mBERT** (`bert-base-multilingual-uncased`) - *Best OOD Generalization*
- **XLM-RoBERTa** (`xlm-roberta-base`)

---

## 📊 Performance Highlights

The system was evaluated on both **In-Domain (ID)** political texts and **Out-of-Domain (OOD)** cultural/revisionist texts.

| Model | Binary F1 (ID) | Binary F1 (OOD) | ROC-AUC | Tactic F1 (ID) | Tactic F1 (OOD) |
|-------|:--------------:|:---------------:|:-------:|:--------------:|:---------------:|
| **RoBERTa-Tagalog** | **0.93** | 0.89 | **0.98** | **0.91** | **0.59** |
| mBERT | 0.91 | **0.91** | 0.97 | 0.83 | 0.50 |
| XLM-RoBERTa | 0.89 | 0.87 | 0.95 | 0.76 | 0.48 |

*Metrics: F1-Score is the harmonic mean of precision and recall. ROC-AUC measures binary classification capability across thresholds. Tactic scores are Macro-Averaged.*

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
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Trained Models

**⚠️ Models are NOT included in this repository (31GB total)**

**Option A: Google Drive** (Recommended)
```
Download from: https://drive.google.com/drive/folders/1TEbQ7TyzqI7wmLDuighQ4CRUO0S-zw6C?usp=sharing

Extract to: model_outputs/
```

**Option B: Train From Scratch**
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
- **Purpose**: Test generalization robustness

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
@mastersthesis{yourname2026taglish,
  title={Detecting Gaslighting Tactics in Taglish Political Discourse},
  author={Your Name},
  year={2026},
  school={Your University}
}
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

## 📈 Detailed Results

### Binary Classification (Detection)
*Determining if text contains gaslighting or not.*
- **Best Model**: RoBERTa-Tagalog
- **In-Domain F1**: 0.93
- **OOD F1**: 0.89
- **ROC-AUC**: 0.98
- **Key Finding**: mBERT showed superior generalization to out-of-domain data (-0.005 ΔF1), indicating it is more robust to topic shifts than RoBERTa.

### Tactic Classification (Identification)
*Identifying specific tactics: Distortion, Trivialization, Coercion, Knowledge Invalidation.*
- **Best Model**: RoBERTa-Tagalog
- **In-Domain Macro-F1**: 0.91
- **OOD Macro-F1**: 0.59
- **Key Finding**: All models struggled with OOD tactic classification (significant performance drop), highlighting the difficulty of mapping specific linguistic manipulation patterns across different cultural contexts.

*Detailed results in `evaluation_results/comprehensive_training_summary.csv`*

---

## 🤝 Contributing

This is an academic research project. For questions or collaboration:
- Open an issue
- Submit a pull request
- Email: jtugado@gbox.adnu.edu.p

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

**Author**: Jude Philippe M. Tugado, Jasper Gomez
**Institution**: Ateneo De Naga University
**Year**: 2026

---

**Last Updated**: February 2026.