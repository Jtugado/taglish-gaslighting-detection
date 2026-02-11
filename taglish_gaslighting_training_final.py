"""
Taglish Gaslighting Detection - Training Script with Comprehensive Evaluation
==============================================================================

This script trains 3 transformer models with comprehensive evaluation metrics:
1. RoBERTa-Tagalog (jcblaise/roberta-tagalog-base) - Monolingual Filipino baseline
2. mBERT (bert-base-multilingual-uncased) - Multilingual baseline  
3. XLM-RoBERTa (xlm-roberta-base) - Strong multilingual baseline

Each model trains TWO classifiers:
- Binary Classifier: Gaslighting vs Non-Gaslighting
- Tactic Classifier: 4 Gaslighting Tactics (for gaslighting texts only)

Evaluation Metrics (Following Thesis Methodology):
==================================================

PRIMARY METRICS:
- Precision, Recall, F1-Score for gaslighting (positive) class
- Macro-F1 across all classes
- Per-tactic Precision, Recall, F1-Score
- Macro-F1 for tactic classification

DIAGNOSTIC METRICS:
- ROC-AUC (binary classification)
- Confusion Matrices per tactic
- Per-class performance metrics

GENERALIZATION METRICS:
- ΔF1 (zero-shot) = F1_in-domain - F1_out-of-domain
  * ΔF1 < 0.10: Robust generalization
  * ΔF1 0.10-0.15: Moderate domain dependence
  * ΔF1 ≥ 0.15: Significant overfitting

ERROR ANALYSIS PREPARATION:
- Detailed misclassification exports for IMT coding
- False Positive/False Negative analysis
- Code-switching type consideration support

Output Directory: D:/Thesis/Taglish_Gaslighting_V3
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Centralized configuration"""
    
    # Hugging Face (REPLACE WITH YOUR TOKEN)
    import os

    # Option A: Best Practice (Reads from your PC's environment)
    token = os.getenv("HF_TOKEN")
    HF_USERNAME = "Tokyosaurus"
    
    # Paths
    BASE_DIR = "D:/Thesis/Taglish_Gaslighting_V3"
    OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
    CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
    ANALYSIS_DIR = os.path.join(BASE_DIR, "error_analysis")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    
    # Data files
    TRAIN_FILE = "training_dataset.csv"
    VAL_FILE = "validation_dataset.csv"
    TEST_IN_DOMAIN = "test_dataset.csv"
    TEST_OOD = "ood_test_dataset.csv"
    
    # Models
    MODELS = {
        "roberta-tagalog": {
            "name": "jcblaise/roberta-tagalog-base",
            "display": "RoBERTa-Tagalog",
            "description": "Monolingual Filipino baseline"
        },
        "mbert": {
            "name": "bert-base-multilingual-uncased",
            "display": "mBERT",
            "description": "Multilingual baseline"
        },
        "xlm-roberta": {
            "name": "xlm-roberta-base",
            "display": "XLM-RoBERTa",
            "description": "Strong multilingual baseline"
        }
    }
    
    # Training hyperparameters
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_LENGTH = 128
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 2
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = True if DEVICE == "cuda" else False
    
    # ΔF1 Thresholds (from methodology)
    DELTA_F1_ROBUST = 0.10      # < 0.10: Robust generalization
    DELTA_F1_MODERATE = 0.15    # 0.10-0.15: Moderate domain dependence
                                # >= 0.15: Significant overfitting
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.RESULTS_DIR, cls.CACHE_DIR, 
                         cls.ANALYSIS_DIR, cls.PLOTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)
        print(f"Device: {cls.DEVICE}")
        if cls.DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed Precision (FP16): {cls.USE_FP16}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Max Sequence Length: {cls.MAX_LENGTH}")
        print(f"\nOutput Directory: {cls.OUTPUT_DIR}")
        print(f"Error Analysis Directory: {cls.ANALYSIS_DIR}")
        print(f"Plots Directory: {cls.PLOTS_DIR}")
        print("\nEVALUATION METRICS:")
        print("  ✓ Precision, Recall, F1 (per-class & macro)")
        print("  ✓ ROC-AUC (binary classification)")
        print("  ✓ Confusion Matrices")
        print("  ✓ ΔF1 (domain shift quantification)")
        print("  ✓ Error Analysis Exports (IMT & Code-Switching)")
        print("="*70 + "\n")

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================

class DataLoader:
    """Handle data loading and preprocessing"""
    
    def __init__(self):
        self.train_df = None
        self.val_df = None
        self.test_id_df = None
        self.test_ood_df = None
    
    def load_datasets(self):
        """Load all datasets"""
        print("\n📂 Loading Datasets...")
        
        # Training data
        if not os.path.exists(Config.TRAIN_FILE):
            raise FileNotFoundError(f"Training file not found: {Config.TRAIN_FILE}")
        self.train_df = pd.read_csv(Config.TRAIN_FILE)
        print(f"  ✓ Training: {len(self.train_df)} samples")
        
        # Validation data
        if not os.path.exists(Config.VAL_FILE):
            raise FileNotFoundError(f"Validation file not found: {Config.VAL_FILE}")
        self.val_df = pd.read_csv(Config.VAL_FILE)
        print(f"  ✓ Validation: {len(self.val_df)} samples")
        
        # In-domain test
        if not os.path.exists(Config.TEST_IN_DOMAIN):
            raise FileNotFoundError(f"In-domain test file not found: {Config.TEST_IN_DOMAIN}")
        self.test_id_df = pd.read_csv(Config.TEST_IN_DOMAIN)
        print(f"  ✓ Test (In-Domain - Political): {len(self.test_id_df)} samples")
        
        # OOD test
        if not os.path.exists(Config.TEST_OOD):
            raise FileNotFoundError(f"OOD test file not found: {Config.TEST_OOD}")
        self.test_ood_df = pd.read_csv(Config.TEST_OOD)
        print(f"  ✓ Test (Out-of-Domain - Cultural & Revisionist): {len(self.test_ood_df)} samples")
        
        # Validate required columns
        self._validate_columns()
        
        return self
    
    def _validate_columns(self):
        """Ensure all required columns exist"""
        required_cols = ['sentence', 'binary_label', 'tactic_label']
        
        for df, name in [(self.train_df, "Training"), (self.val_df, "Validation"),
                         (self.test_id_df, "In-Domain Test"), (self.test_ood_df, "OOD Test")]:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"{name} dataset missing columns: {missing}")
    
    def prepare_labels(self):
        """Prepare binary and tactic labels"""
        print("\n🏷️  Preparing Labels...")
        
        # Binary labels (0=Non-Gaslighting, 1=Gaslighting)
        for df in [self.train_df, self.val_df, self.test_id_df, self.test_ood_df]:
            df['label_binary'] = df['binary_label'].apply(
                lambda x: 1 if str(x).lower().strip() == 'gaslighting' else 0
            )
        
        # Tactic labels (0-3 for the 4 gaslighting tactics)
        tactic_mapping = {
            'distortion_denial': 0,
            'trivialization_minimization': 1,
            'coercion_intimidation': 2,
            'knowledge_invalidation': 3
        }
        
        for df in [self.train_df, self.val_df, self.test_id_df, self.test_ood_df]:
            df['tactic_clean'] = df['tactic_label'].astype(str).str.lower().str.strip()
            df['label_tactic'] = df['tactic_clean'].map(tactic_mapping)
        
        # Print label distributions
        print("\n  Binary Distribution (Training - Political Domain):")
        print(f"    Non-Gaslighting: {(self.train_df['label_binary']==0).sum()}")
        print(f"    Gaslighting: {(self.train_df['label_binary']==1).sum()}")
        
        print("\n  Tactic Distribution (Training - Gaslighting only):")
        tactic_names = {
            0: 'Distortion & Denial',
            1: 'Trivialization & Minimization', 
            2: 'Coercion & Intimidation',
            3: 'Knowledge Invalidation'
        }
        gaslighting_train = self.train_df[self.train_df['label_binary'] == 1]
        for label_id, name in tactic_names.items():
            count = (gaslighting_train['label_tactic']==label_id).sum()
            print(f"    {name}: {count}")
        
        return self
    
    def get_datasets_binary(self):
        """Get datasets for binary classification"""
        cols_to_keep = ['sentence', 'label_binary']
        
        train_ds = Dataset.from_pandas(
            self.train_df[cols_to_keep].rename(columns={'label_binary': 'label'})
        )
        val_ds = Dataset.from_pandas(
            self.val_df[cols_to_keep].rename(columns={'label_binary': 'label'})
        )
        test_id_ds = Dataset.from_pandas(
            self.test_id_df[cols_to_keep].rename(columns={'label_binary': 'label'})
        )
        test_ood_ds = Dataset.from_pandas(
            self.test_ood_df[cols_to_keep].rename(columns={'label_binary': 'label'})
        )
        
        return {
            'train': train_ds,
            'val': val_ds,
            'test_id': test_id_ds,
            'test_ood': test_ood_ds
        }
    
    def get_datasets_tactic(self):
        """Get datasets for tactic classification (gaslighting only)"""
        
        # Filter only gaslighting samples
        train_gas = self.train_df[self.train_df['label_binary'] == 1].copy()
        val_gas = self.val_df[self.val_df['label_binary'] == 1].copy()
        test_id_gas = self.test_id_df[self.test_id_df['label_binary'] == 1].copy()
        test_ood_gas = self.test_ood_df[self.test_ood_df['label_binary'] == 1].copy()
        
        # Remove NaN tactic labels
        train_gas = train_gas.dropna(subset=['label_tactic'])
        val_gas = val_gas.dropna(subset=['label_tactic'])
        test_id_gas = test_id_gas.dropna(subset=['label_tactic'])
        test_ood_gas = test_ood_gas.dropna(subset=['label_tactic'])
        
        # Convert to int
        for df in [train_gas, val_gas, test_id_gas, test_ood_gas]:
            df['label_tactic'] = df['label_tactic'].astype(int)
        
        cols_to_keep = ['sentence', 'label_tactic']
        
        train_ds = Dataset.from_pandas(
            train_gas[cols_to_keep].rename(columns={'label_tactic': 'label'})
        )
        val_ds = Dataset.from_pandas(
            val_gas[cols_to_keep].rename(columns={'label_tactic': 'label'})
        )
        test_id_ds = Dataset.from_pandas(
            test_id_gas[cols_to_keep].rename(columns={'label_tactic': 'label'})
        )
        test_ood_ds = Dataset.from_pandas(
            test_ood_gas[cols_to_keep].rename(columns={'label_tactic': 'label'})
        )
        
        print(f"\n  Tactic Dataset Sizes (Gaslighting only):")
        print(f"    Training: {len(train_ds)} samples")
        print(f"    Validation: {len(val_ds)} samples")
        print(f"    Test In-Domain: {len(test_id_ds)} samples")
        print(f"    Test OOD: {len(test_ood_ds)} samples")
        
        return {
            'train': train_ds,
            'val': val_ds,
            'test_id': test_id_ds,
            'test_ood': test_ood_ds
        }

# ==========================================
# ENHANCED METRICS COMPUTATION
# ==========================================

def compute_metrics_binary(eval_pred):
    """Compute comprehensive metrics for binary classification"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro-F1
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Gaslighting (positive class) metrics
    gaslighting_precision = precision[1]
    gaslighting_recall = recall[1]
    gaslighting_f1 = f1[1]
    
    # ROC-AUC (using probability of positive class)
    try:
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
    except:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': gaslighting_precision,  # Gaslighting class
        'recall': gaslighting_recall,
        'f1': gaslighting_f1,
        'macro_f1': macro_f1,
        'roc_auc': roc_auc,
        # Per-class for reference
        'non_gas_precision': precision[0],
        'non_gas_recall': recall[0],
        'non_gas_f1': f1[0]
    }

def compute_metrics_tactic(eval_pred):
    """Compute comprehensive metrics for tactic classification"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Per-tactic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro-F1 (equal weight to all tactics)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Weighted F1 (account for class imbalance)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        # Per-tactic metrics
        'distortion_precision': precision[0] if len(precision) > 0 else 0,
        'distortion_recall': recall[0] if len(recall) > 0 else 0,
        'distortion_f1': f1[0] if len(f1) > 0 else 0,
        'trivialization_precision': precision[1] if len(precision) > 1 else 0,
        'trivialization_recall': recall[1] if len(recall) > 1 else 0,
        'trivialization_f1': f1[1] if len(f1) > 1 else 0,
        'coercion_precision': precision[2] if len(precision) > 2 else 0,
        'coercion_recall': recall[2] if len(recall) > 2 else 0,
        'coercion_f1': f1[2] if len(f1) > 2 else 0,
        'knowledge_precision': precision[3] if len(precision) > 3 else 0,
        'knowledge_recall': recall[3] if len(recall) > 3 else 0,
        'knowledge_f1': f1[3] if len(f1) > 3 else 0
    }

# ==========================================
# VISUALIZATION & ANALYSIS
# ==========================================

class ResultsVisualizer:
    """Generate plots and analysis outputs"""
    
    @staticmethod
    def plot_confusion_matrix(cm, labels, title, save_path):
        """Generate and save confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc, title, save_path):
        """Generate and save ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def export_misclassifications(y_true, y_pred, sentences, labels_map, 
                                  save_path, split_name):
        """Export misclassified samples for IMT coding and error analysis"""
        
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print(f"    No misclassifications in {split_name}")
            return
        
        error_data = []
        for idx in misclassified_indices:
            error_data.append({
                'sentence': sentences[idx],
                'true_label': labels_map[y_true[idx]],
                'predicted_label': labels_map[y_pred[idx]],
                'true_label_id': int(y_true[idx]),
                'predicted_label_id': int(y_pred[idx]),
                'error_type': 'False Positive' if y_pred[idx] > y_true[idx] else 'False Negative',
                # Placeholders for manual IMT coding
                'imt_quality_violation': '',
                'imt_quantity_violation': '',
                'imt_relevance_violation': '',
                'imt_manner_violation': '',
                # Placeholders for code-switching analysis
                'code_switching_type': '',  # inter-sentential / intra-sentential
                'code_switching_pattern': '',
                # Analysis notes
                'analysis_notes': ''
            })
        
        error_df = pd.DataFrame(error_data)
        error_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"    Exported {len(error_data)} misclassifications to {save_path}")
        
        # Print summary
        false_positives = (error_df['error_type'] == 'False Positive').sum()
        false_negatives = (error_df['error_type'] == 'False Negative').sum()
        print(f"      False Positives: {false_positives}")
        print(f"      False Negatives: {false_negatives}")

# ==========================================
# MODEL TRAINING WITH ENHANCED EVALUATION
# ==========================================

class SequentialModelTrainer:
    """Train and comprehensively evaluate models"""
    
    def __init__(self, model_key, data_loader):
        self.model_key = model_key
        self.model_config = Config.MODELS[model_key]
        self.model_name = self.model_config['name']
        self.data_loader = data_loader
        
        # Output paths
        self.output_dir_binary = os.path.join(Config.OUTPUT_DIR, f"{model_key}_binary")
        self.output_dir_tactic = os.path.join(Config.OUTPUT_DIR, f"{model_key}_tactic")
        
        os.makedirs(self.output_dir_binary, exist_ok=True)
        os.makedirs(self.output_dir_tactic, exist_ok=True)
        
        self.results = {}
    
    def train_binary_classifier(self):
        """Train binary classifier with comprehensive evaluation"""
        print(f"\n{'='*70}")
        print(f"TRAINING BINARY CLASSIFIER: {self.model_config['display']}")
        print(f"{'='*70}")
        
        # Get datasets
        datasets = self.data_loader.get_datasets_binary()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=Config.CACHE_DIR
        )
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['sentence'],
                padding='max_length',
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
        
        print("\n🔤 Tokenizing datasets...")
        tokenized_datasets = {}
        original_sentences = {}  # Store for error analysis
        
        for split_name, dataset in datasets.items():
            # Store original sentences before tokenization
            original_sentences[split_name] = dataset['sentence']
            
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                desc=f"Tokenizing {split_name}"
            )
            tokenized_datasets[split_name].set_format(
                'torch',
                columns=['input_ids', 'attention_mask', 'label']
            )
        
        # Load model
        print(f"\n🤖 Loading {self.model_config['display']} model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "Non-Gaslighting", 1: "Gaslighting"},
            label2id={"Non-Gaslighting": 0, "Gaslighting": 1},
            cache_dir=Config.CACHE_DIR
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir_binary,
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.NUM_EPOCHS,
            weight_decay=Config.WEIGHT_DECAY,
            warmup_ratio=Config.WARMUP_RATIO,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=Config.USE_FP16,
            dataloader_num_workers=2,
            save_total_limit=2,
            logging_dir=os.path.join(self.output_dir_binary, 'logs'),
            logging_strategy="epoch",
            report_to="none",
            seed=42,
            push_to_hub=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics_binary,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.EARLY_STOPPING_PATIENCE)]
        )
        
        # Train
        print("\n⏳ Training binary classifier...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model(self.output_dir_binary)
        tokenizer.save_pretrained(self.output_dir_binary)
        
        print(f"\n✅ Binary classifier trained!")
        print(f"📊 Training Loss: {train_result.training_loss:.4f}")
        
        # Comprehensive evaluation
        print(f"\n📊 Comprehensive Evaluation...")
        results_binary = self._evaluate_binary_model(
            trainer, tokenized_datasets, original_sentences
        )
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        
        return results_binary
    
    def _evaluate_binary_model(self, trainer, tokenized_datasets, original_sentences):
        """Comprehensive binary model evaluation"""
        results = {}
        
        id2label = {0: "Non-Gaslighting", 1: "Gaslighting"}
        
        for split_name, split_label in [('val', 'Validation'),
                                        ('test_id', 'In-Domain Test (Political)'),
                                        ('test_ood', 'Out-of-Domain Test (Cultural & Revisionist)')]:
            
            print(f"\n  • {split_label}")
            
            # Get predictions
            predictions = trainer.predict(tokenized_datasets[split_name])
            y_true = predictions.label_ids
            y_pred = np.argmax(predictions.predictions, axis=-1)
            y_prob = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
            
            # Compute metrics
            eval_results = trainer.evaluate(tokenized_datasets[split_name])
            results[split_name] = self._format_results(eval_results)
            
            # Print metrics
            self._print_binary_metrics(results[split_name])
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            results[split_name]['confusion_matrix'] = cm.tolist()
            
            # Save confusion matrix plot
            cm_plot_path = os.path.join(
                Config.PLOTS_DIR,
                f'{self.model_key}_binary_cm_{split_name}.png'
            )
            ResultsVisualizer.plot_confusion_matrix(
                cm, list(id2label.values()),
                f'{self.model_config["display"]} - Binary Classification\n{split_label}',
                cm_plot_path
            )
            print(f"      Confusion Matrix saved: {cm_plot_path}")
            
            # ROC Curve (for test sets)
            if 'test' in split_name:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc = results[split_name].get('roc_auc', 0)
                
                roc_plot_path = os.path.join(
                    Config.PLOTS_DIR,
                    f'{self.model_key}_binary_roc_{split_name}.png'
                )
                ResultsVisualizer.plot_roc_curve(
                    fpr, tpr, roc_auc,
                    f'{self.model_config["display"]} - ROC Curve\n{split_label}',
                    roc_plot_path
                )
                print(f"      ROC Curve saved: {roc_plot_path}")
            
            # Error Analysis Export
            error_export_path = os.path.join(
                Config.ANALYSIS_DIR,
                f'{self.model_key}_binary_errors_{split_name}.csv'
            )
            ResultsVisualizer.export_misclassifications(
                y_true, y_pred, original_sentences[split_name],
                id2label, error_export_path, split_label
            )
        
        # Calculate ΔF1 (domain shift metric)
        delta_f1 = results['test_id']['f1'] - results['test_ood']['f1']
        results['delta_f1'] = delta_f1
        
        # Interpret ΔF1
        if delta_f1 < Config.DELTA_F1_ROBUST:
            interpretation = "✅ Robust generalization (minimal domain sensitivity)"
        elif delta_f1 < Config.DELTA_F1_MODERATE:
            interpretation = "⚠️ Moderate domain dependence"
        else:
            interpretation = "❌ Significant overfitting (domain adaptation needed)"
        
        results['delta_f1_interpretation'] = interpretation
        
        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  DOMAIN GENERALIZATION ANALYSIS (ΔF1)")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  In-Domain F1:  {results['test_id']['f1']:.4f}")
        print(f"  Out-of-Domain F1: {results['test_ood']['f1']:.4f}")
        print(f"  ΔF1 = {delta_f1:.4f}")
        print(f"  {interpretation}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Save results
        self._save_results(results, self.output_dir_binary, 'binary')
        
        return results
    
    def train_tactic_classifier(self):
        """Train tactic classifier with comprehensive evaluation"""
        print(f"\n{'='*70}")
        print(f"TRAINING TACTIC CLASSIFIER: {self.model_config['display']}")
        print(f"{'='*70}")
        
        # Get datasets (gaslighting only)
        datasets = self.data_loader.get_datasets_tactic()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=Config.CACHE_DIR
        )
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['sentence'],
                padding='max_length',
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
        
        print("\n🔤 Tokenizing datasets...")
        tokenized_datasets = {}
        original_sentences = {}
        
        for split_name, dataset in datasets.items():
            original_sentences[split_name] = dataset['sentence']
            
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                desc=f"Tokenizing {split_name}"
            )
            tokenized_datasets[split_name].set_format(
                'torch',
                columns=['input_ids', 'attention_mask', 'label']
            )
        
        # Load model
        print(f"\n🤖 Loading {self.model_config['display']} model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=4,
            id2label={
                0: "Distortion & Denial",
                1: "Trivialization & Minimization",
                2: "Coercion & Intimidation",
                3: "Knowledge Invalidation"
            },
            label2id={
                "Distortion & Denial": 0,
                "Trivialization & Minimization": 1,
                "Coercion & Intimidation": 2,
                "Knowledge Invalidation": 3
            },
            cache_dir=Config.CACHE_DIR
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir_tactic,
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.NUM_EPOCHS,
            weight_decay=Config.WEIGHT_DECAY,
            warmup_ratio=Config.WARMUP_RATIO,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=Config.USE_FP16,
            dataloader_num_workers=2,
            save_total_limit=2,
            logging_dir=os.path.join(self.output_dir_tactic, 'logs'),
            logging_strategy="epoch",
            report_to="none",
            seed=42,
            push_to_hub=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics_tactic,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.EARLY_STOPPING_PATIENCE)]
        )
        
        # Train
        print("\n⏳ Training tactic classifier...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model(self.output_dir_tactic)
        tokenizer.save_pretrained(self.output_dir_tactic)
        
        print(f"\n✅ Tactic classifier trained!")
        print(f"📊 Training Loss: {train_result.training_loss:.4f}")
        
        # Comprehensive evaluation
        print(f"\n📊 Comprehensive Evaluation...")
        results_tactic = self._evaluate_tactic_model(
            trainer, tokenized_datasets, original_sentences
        )
        
        # Cleanup
        del model, trainer
        torch.cuda.empty_cache()
        
        return results_tactic
    
    def _evaluate_tactic_model(self, trainer, tokenized_datasets, original_sentences):
        """Comprehensive tactic model evaluation"""
        results = {}
        
        id2label = {
            0: "Distortion & Denial",
            1: "Trivialization & Minimization",
            2: "Coercion & Intimidation",
            3: "Knowledge Invalidation"
        }
        
        for split_name, split_label in [('val', 'Validation'),
                                        ('test_id', 'In-Domain Test (Political)'),
                                        ('test_ood', 'Out-of-Domain Test (Cultural & Revisionist)')]:
            
            print(f"\n  • {split_label}")
            
            # Get predictions
            predictions = trainer.predict(tokenized_datasets[split_name])
            y_true = predictions.label_ids
            y_pred = np.argmax(predictions.predictions, axis=-1)
            
            # Compute metrics
            eval_results = trainer.evaluate(tokenized_datasets[split_name])
            results[split_name] = self._format_results(eval_results)
            
            # Print metrics
            self._print_tactic_metrics(results[split_name])
            
            # Detailed per-tactic report
            report = classification_report(
                y_true, y_pred,
                target_names=list(id2label.values()),
                digits=4,
                zero_division=0
            )
            
            report_file = os.path.join(
                self.output_dir_tactic,
                f'tactic_classification_report_{split_name}.txt'
            )
            with open(report_file, 'w') as f:
                f.write(f"Per-Tactic Classification Report - {split_label}\n")
                f.write("="*70 + "\n\n")
                f.write(report)
            
            print(f"      Detailed report saved: {report_file}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            results[split_name]['confusion_matrix'] = cm.tolist()
            
            # Save confusion matrix plot
            cm_plot_path = os.path.join(
                Config.PLOTS_DIR,
                f'{self.model_key}_tactic_cm_{split_name}.png'
            )
            ResultsVisualizer.plot_confusion_matrix(
                cm, list(id2label.values()),
                f'{self.model_config["display"]} - Tactic Classification\n{split_label}',
                cm_plot_path
            )
            print(f"      Confusion Matrix saved: {cm_plot_path}")
            
            # Error Analysis Export
            error_export_path = os.path.join(
                Config.ANALYSIS_DIR,
                f'{self.model_key}_tactic_errors_{split_name}.csv'
            )
            ResultsVisualizer.export_misclassifications(
                y_true, y_pred, original_sentences[split_name],
                id2label, error_export_path, split_label
            )
        
        # Calculate ΔF1 (domain shift metric) - using macro_f1
        delta_f1 = results['test_id']['macro_f1'] - results['test_ood']['macro_f1']
        results['delta_f1'] = delta_f1
        
        # Interpret ΔF1
        if delta_f1 < Config.DELTA_F1_ROBUST:
            interpretation = "✅ Robust generalization (minimal domain sensitivity)"
        elif delta_f1 < Config.DELTA_F1_MODERATE:
            interpretation = "⚠️ Moderate domain dependence"
        else:
            interpretation = "❌ Significant overfitting (domain adaptation needed)"
        
        results['delta_f1_interpretation'] = interpretation
        
        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  DOMAIN GENERALIZATION ANALYSIS (ΔF1 - Macro)")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  In-Domain Macro-F1:  {results['test_id']['macro_f1']:.4f}")
        print(f"  Out-of-Domain Macro-F1: {results['test_ood']['macro_f1']:.4f}")
        print(f"  ΔF1 = {delta_f1:.4f}")
        print(f"  {interpretation}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Save results
        self._save_results(results, self.output_dir_tactic, 'tactic')
        
        return results
    
    def _format_results(self, raw_results):
        """Format results dict"""
        formatted = {}
        for key, value in raw_results.items():
            if key.startswith('eval_'):
                formatted[key.replace('eval_', '')] = value
        return formatted
    
    def _print_binary_metrics(self, metrics):
        """Print binary classification metrics"""
        print(f"      Accuracy:       {metrics.get('accuracy', 0):.4f}")
        print(f"      Precision (Gas):{metrics.get('precision', 0):.4f}")
        print(f"      Recall (Gas):   {metrics.get('recall', 0):.4f}")
        print(f"      F1 (Gas):       {metrics.get('f1', 0):.4f}")
        print(f"      Macro-F1:       {metrics.get('macro_f1', 0):.4f}")
        print(f"      ROC-AUC:        {metrics.get('roc_auc', 0):.4f}")
    
    def _print_tactic_metrics(self, metrics):
        """Print tactic classification metrics"""
        print(f"      Accuracy:       {metrics.get('accuracy', 0):.4f}")
        print(f"      Macro-F1:       {metrics.get('macro_f1', 0):.4f}")
        print(f"      Weighted-F1:    {metrics.get('weighted_f1', 0):.4f}")
        print(f"\n      Per-Tactic F1 Scores:")
        print(f"        Distortion & Denial:       {metrics.get('distortion_f1', 0):.4f}")
        print(f"        Trivialization & Min.:     {metrics.get('trivialization_f1', 0):.4f}")
        print(f"        Coercion & Intimidation:   {metrics.get('coercion_f1', 0):.4f}")
        print(f"        Knowledge Invalidation:    {metrics.get('knowledge_f1', 0):.4f}")
    
    def _save_results(self, results, output_dir, task):
        """Save comprehensive results"""
        # Remove non-serializable items
        results_to_save = {}
        for split, metrics in results.items():
            if split not in ['delta_f1', 'delta_f1_interpretation']:
                results_to_save[split] = {
                    k: v for k, v in metrics.items() 
                    if k not in ['predictions']
                }
        
        # Add ΔF1 metrics
        if 'delta_f1' in results:
            results_to_save['delta_f1'] = results['delta_f1']
            results_to_save['delta_f1_interpretation'] = results['delta_f1_interpretation']
        
        results_file = os.path.join(output_dir, f'comprehensive_results_{task}.json')
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n  💾 Results saved to: {results_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("TAGLISH GASLIGHTING DETECTION - COMPREHENSIVE TRAINING & EVALUATION")
    print("="*70)
    
    # Setup
    Config.setup_directories()
    Config.print_config()
    
    # Load data
    data_loader = DataLoader()
    data_loader.load_datasets()
    data_loader.prepare_labels()
    
    # Results summary
    all_results = []
    
    # Train all models
    for model_key in Config.MODELS.keys():
        
        print("\n" + "="*70)
        print(f"MODEL: {Config.MODELS[model_key]['display']}")
        print("="*70)
        
        try:
            # Initialize trainer
            trainer = SequentialModelTrainer(model_key, data_loader)
            
            # Train binary classifier
            results_binary = trainer.train_binary_classifier()
            
            # Train tactic classifier
            results_tactic = trainer.train_tactic_classifier()
            
            # Store results
            all_results.append({
                'model': Config.MODELS[model_key]['display'],
                'model_key': model_key,
                
                # Binary results
                'binary_val_f1': results_binary['val']['f1'],
                'binary_test_id_f1': results_binary['test_id']['f1'],
                'binary_test_ood_f1': results_binary['test_ood']['f1'],
                'binary_roc_auc_id': results_binary['test_id']['roc_auc'],
                'binary_roc_auc_ood': results_binary['test_ood']['roc_auc'],
                'binary_delta_f1': results_binary['delta_f1'],
                
                # Tactic results
                'tactic_val_macro_f1': results_tactic['val']['macro_f1'],
                'tactic_test_id_macro_f1': results_tactic['test_id']['macro_f1'],
                'tactic_test_ood_macro_f1': results_tactic['test_ood']['macro_f1'],
                'tactic_delta_f1': results_tactic['delta_f1'],
                
                'status': '✅ Success'
            })
            
            # Cleanup
            del trainer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Error training {model_key}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                'model': Config.MODELS[model_key]['display'],
                'model_key': model_key,
                'binary_val_f1': 0.0,
                'binary_test_id_f1': 0.0,
                'binary_test_ood_f1': 0.0,
                'binary_roc_auc_id': 0.0,
                'binary_roc_auc_ood': 0.0,
                'binary_delta_f1': 0.0,
                'tactic_val_macro_f1': 0.0,
                'tactic_test_id_macro_f1': 0.0,
                'tactic_test_ood_macro_f1': 0.0,
                'tactic_delta_f1': 0.0,
                'status': f'❌ Failed: {str(e)[:50]}'
            })
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL COMPREHENSIVE RESULTS")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    # Binary Classification Summary
    print("\n" + "─"*70)
    print("BINARY CLASSIFICATION (Gaslighting Detection)")
    print("─"*70)
    binary_cols = ['model', 'binary_test_id_f1', 'binary_test_ood_f1', 
                   'binary_roc_auc_id', 'binary_delta_f1']
    binary_display = results_df[binary_cols].copy()
    binary_display.columns = ['Model', 'In-Domain F1', 'OOD F1', 'ROC-AUC (ID)', 'ΔF1']
    print(binary_display.to_string(index=False))
    
    # Tactic Classification Summary
    print("\n" + "─"*70)
    print("TACTIC CLASSIFICATION (4-Class)")
    print("─"*70)
    tactic_cols = ['model', 'tactic_test_id_macro_f1', 'tactic_test_ood_macro_f1', 
                   'tactic_delta_f1']
    tactic_display = results_df[tactic_cols].copy()
    tactic_display.columns = ['Model', 'In-Domain Macro-F1', 'OOD Macro-F1', 'ΔF1']
    print(tactic_display.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(Config.RESULTS_DIR, 'comprehensive_training_summary.csv')
    results_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Generate thesis tables
    generate_thesis_tables(results_df)
    
    print("\n" + "="*70)
    print("🎉 ALL TRAINING & EVALUATION COMPLETED!")
    print("="*70)
    print(f"\n📂 Output Locations:")
    print(f"   Models:           {Config.OUTPUT_DIR}")
    print(f"   Results:          {Config.RESULTS_DIR}")
    print(f"   Error Analysis:   {Config.ANALYSIS_DIR}")
    print(f"   Plots:            {Config.PLOTS_DIR}")
    print("\n📊 Generated Artifacts:")
    print("   ✓ Trained models (binary & tactic for each)")
    print("   ✓ Confusion matrices (plots & CSV)")
    print("   ✓ ROC curves")
    print("   ✓ Classification reports (per-tactic metrics)")
    print("   ✓ Error analysis exports (for IMT coding)")
    print("   ✓ ΔF1 domain shift analysis")
    print("="*70)

def generate_thesis_tables(results_df):
    """Generate formatted tables for thesis"""
    
    print("\n" + "="*70)
    print("THESIS TABLES (Following Methodology Section 4.4 & 4.5)")
    print("="*70)
    
    # Table 1: Binary Classification with ROC-AUC and ΔF1
    print("\nTable 1: Binary Classification Performance with Domain Generalization")
    print("─"*70)
    binary_table = results_df[[
        'model', 
        'binary_test_id_f1', 
        'binary_roc_auc_id',
        'binary_test_ood_f1',
        'binary_roc_auc_ood',
        'binary_delta_f1'
    ]].copy()
    binary_table.columns = [
        'Model', 
        'In-Domain F1', 
        'In-Domain ROC-AUC',
        'OOD F1',
        'OOD ROC-AUC',
        'ΔF1'
    ]
    print(binary_table.to_string(index=False))
    
    # Table 2: Tactic Classification with ΔF1
    print("\nTable 2: Tactic Classification Performance (Macro-F1) with Domain Generalization")
    print("─"*70)
    tactic_table = results_df[[
        'model',
        'tactic_test_id_macro_f1',
        'tactic_test_ood_macro_f1',
        'tactic_delta_f1'
    ]].copy()
    tactic_table.columns = [
        'Model',
        'In-Domain Macro-F1',
        'OOD Macro-F1',
        'ΔF1'
    ]
    print(tactic_table.to_string(index=False))
    
    # Save tables
    binary_table.to_csv(
        os.path.join(Config.RESULTS_DIR, 'thesis_table1_binary_with_metrics.csv'),
        index=False
    )
    tactic_table.to_csv(
        os.path.join(Config.RESULTS_DIR, 'thesis_table2_tactic_with_metrics.csv'),
        index=False
    )
    
    # Generate ΔF1 interpretation summary
    print("\nΔF1 Interpretation Guide:")
    print("  ΔF1 < 0.10:  ✅ Robust generalization (minimal domain sensitivity)")
    print("  ΔF1 0.10-0.15: ⚠️ Moderate domain dependence")
    print("  ΔF1 ≥ 0.15:  ❌ Significant overfitting (domain adaptation needed)")
    
    print(f"\n💾 Thesis tables saved to: {Config.RESULTS_DIR}")
    print("\n📝 Additional outputs for thesis:")
    print(f"   • Per-tactic performance: See classification_report_*.txt files")
    print(f"   • Confusion matrices: See {Config.PLOTS_DIR}")
    print(f"   • Error analysis (IMT coding): See {Config.ANALYSIS_DIR}")

if __name__ == "__main__":
    main()