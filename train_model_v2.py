"""
Taglish Gaslighting Detection - Model Training Script
======================================================

This script trains and evaluates three transformer models for gaslighting detection:
1. RoBERTa-Tagalog (jcblaise/roberta-tagalog-base) - Monolingual Filipino baseline
2. mBERT (bert-base-multilingual-uncased) - Multilingual baseline  
3. XLM-RoBERTa (xlm-roberta-base) - Strong multilingual baseline

Tasks:
- Binary Classification: Gaslighting vs Non-Gaslighting (2 classes)
- Multi-class Classification: 4 Gaslighting Tactics + Non-Gaslighting (5 classes total)
  * Class 0: Non-Gaslighting
  * Class 1: Distortion & Denial
  * Class 2: Trivialization & Minimization
  * Class 3: Coercion & Intimidation
  * Class 4: Knowledge Invalidation

Evaluation:
- In-domain test set (political domain)
- Out-of-domain test set (cultural & revisionist domains)

Output Directory: D:/Thesis/Taglish_Gaslighting_V2
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
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Centralized configuration"""
    
    import os

    # Option A: Best Practice (Reads from your PC's environment)
    token = os.getenv("HF_TOKEN")
    HF_USERNAME = "Tokyosaurus"
    
    # Paths - OUTPUT TO D: DRIVE
    BASE_DIR = "D:/Thesis/Taglish_Gaslighting_V2"
    OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
    CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
    
    # Data files
    TRAIN_FILE = "training_dataset.csv"
    VAL_FILE = "validation_dataset.csv"
    TEST_IN_DOMAIN = "test_dataset.csv"
    TEST_OOD = "ood_test_dataset.csv"
    
    # Models - aligned with thesis
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
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.OUTPUT_DIR, cls.RESULTS_DIR, cls.CACHE_DIR]:
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
        print(f"Output Directory: {cls.OUTPUT_DIR}")
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
        print(f"  ✓ Test (In-Domain): {len(self.test_id_df)} samples")
        
        # OOD test
        if not os.path.exists(Config.TEST_OOD):
            raise FileNotFoundError(f"OOD test file not found: {Config.TEST_OOD}")
        self.test_ood_df = pd.read_csv(Config.TEST_OOD)
        print(f"  ✓ Test (OOD): {len(self.test_ood_df)} samples")
        
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
        """Prepare binary and multi-class labels"""
        print("\n🏷️  Preparing Labels...")
        
        # Binary labels
        for df in [self.train_df, self.val_df, self.test_id_df, self.test_ood_df]:
            df['label_binary'] = df['binary_label'].apply(
                lambda x: 1 if str(x).lower().strip() == 'gaslighting' else 0
            )
        
        # Tactic labels (multi-class)
        # IMPORTANT: We have 4 gaslighting tactics + 1 non-gaslighting class = 5 total classes
        # Labels 1-4 are the 4 gaslighting tactics
        # Label 0 is non-gaslighting (which includes both "non_gaslighting" and "na" from data)
        tactic_mapping = {
            'non_gaslighting': 0,
            'na': 0,  # NA is just another label for non-gaslighting
            'distortion_denial': 1,
            'trivialization_minimization': 2,
            'coercion_intimidation': 3,
            'knowledge_invalidation': 4
        }
        
        for df in [self.train_df, self.val_df, self.test_id_df, self.test_ood_df]:
            df['tactic_clean'] = df['tactic_label'].astype(str).str.lower().str.strip()
            df['label_tactic'] = df['tactic_clean'].map(tactic_mapping).fillna(0).astype(int)
        
        # Print label distributions
        print("\n  Binary Distribution (Training):")
        print(f"    Non-Gaslighting: {(self.train_df['label_binary']==0).sum()}")
        print(f"    Gaslighting: {(self.train_df['label_binary']==1).sum()}")
        
        print("\n  Tactic Distribution (Training):")
        tactic_names = {0: 'Non-Gaslighting', 1: 'Distortion', 2: 'Trivialization', 
                       3: 'Coercion', 4: 'Knowledge Inv.'}
        for label_id, name in tactic_names.items():
            count = (self.train_df['label_tactic']==label_id).sum()
            print(f"    {name}: {count}")
        
        return self
    
    def get_datasets(self, task='binary'):
        """
        Convert to HuggingFace Dataset format
        
        Args:
            task: 'binary' or 'tactic'
        
        Returns:
            Dictionary with train, val, test_id, test_ood datasets
        """
        label_col = 'label_binary' if task == 'binary' else 'label_tactic'
        
        # Keep only necessary columns
        cols_to_keep = ['sentence', label_col]
        
        train_ds = Dataset.from_pandas(
            self.train_df[cols_to_keep].rename(columns={label_col: 'label'})
        )
        val_ds = Dataset.from_pandas(
            self.val_df[cols_to_keep].rename(columns={label_col: 'label'})
        )
        test_id_ds = Dataset.from_pandas(
            self.test_id_df[cols_to_keep].rename(columns={label_col: 'label'})
        )
        test_ood_ds = Dataset.from_pandas(
            self.test_ood_df[cols_to_keep].rename(columns={label_col: 'label'})
        )
        
        return {
            'train': train_ds,
            'val': val_ds,
            'test_id': test_id_ds,
            'test_ood': test_ood_ds
        }

# ==========================================
# METRICS COMPUTATION
# ==========================================

def compute_metrics(eval_pred, task='binary'):
    """
    Compute comprehensive metrics
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        task: 'binary' or 'tactic'
    
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Determine averaging strategy
    if task == 'binary':
        average = 'binary'
    else:
        average = 'weighted'  # For multi-class
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    # Macro F1 for multi-class (important for imbalanced classes)
    if task == 'tactic':
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_f1': macro_f1
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ==========================================
# MODEL TRAINING
# ==========================================

class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, model_key, task, datasets):
        """
        Initialize trainer
        
        Args:
            model_key: Key from Config.MODELS dict
            task: 'binary' or 'tactic'
            datasets: Dict with train/val/test datasets
        """
        self.model_key = model_key
        self.model_config = Config.MODELS[model_key]
        self.model_name = self.model_config['name']
        self.task = task
        self.datasets = datasets
        
        # Label mappings
        if task == 'binary':
            self.num_labels = 2
            self.id2label = {0: "Non-Gaslighting", 1: "Gaslighting"}
        else:
            # Multi-class: 5 classes (0=Non-Gas, 1-4=4 Tactics)
            self.num_labels = 5
            self.id2label = {
                0: "Non-Gaslighting",
                1: "Distortion & Denial",
                2: "Trivialization & Minimization",
                3: "Coercion & Intimidation",
                4: "Knowledge Invalidation"
            }
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Output paths
        self.output_dir = os.path.join(
            Config.OUTPUT_DIR, 
            f"{task}_{model_key}"
        )
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(self):
        """Tokenize datasets"""
        print(f"\n🔤 Tokenizing with {self.model_config['display']}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=Config.CACHE_DIR
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['sentence'],
                padding='max_length',
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
        
        # Tokenize all datasets
        self.tokenized_datasets = {}
        for split_name, dataset in self.datasets.items():
            self.tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                desc=f"Tokenizing {split_name}"
            )
            
            # Set format
            self.tokenized_datasets[split_name].set_format(
                'torch',
                columns=['input_ids', 'attention_mask', 'label']
            )
        
        print("  ✓ Tokenization complete")
    
    def train(self):
        """Train the model"""
        print(f"\n🚀 Training {self.model_config['display']} for {self.task} task...")
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            cache_dir=Config.CACHE_DIR
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            
            # Optimization
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.NUM_EPOCHS,
            weight_decay=Config.WEIGHT_DECAY,
            warmup_ratio=Config.WARMUP_RATIO,
            
            # Evaluation
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            
            # Efficiency
            fp16=Config.USE_FP16,
            dataloader_num_workers=2,
            save_total_limit=2,  # Keep only 2 best checkpoints
            
            # Logging
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_strategy="epoch",
            report_to="none",  # Disable wandb/tensorboard
            
            # Other
            seed=42,
            push_to_hub=False  # Set to True if you want to push to HF Hub
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['val'],
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=lambda x: compute_metrics(x, self.task),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.EARLY_STOPPING_PATIENCE)]
        )
        
        # Train
        print("  ⏳ Training started...")
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"  ✅ Training complete!")
        print(f"  📊 Training Loss: {train_result.training_loss:.4f}")
        
        return train_result
    
    def evaluate(self):
        """Evaluate on all test sets"""
        print(f"\n📊 Evaluating {self.model_config['display']} ({self.task})...")
        
        results = {}
        
        # Validation set
        print("  • Validation Set")
        val_results = self.trainer.evaluate(self.tokenized_datasets['val'])
        results['validation'] = self._format_results(val_results)
        self._print_metrics(results['validation'])
        
        # In-domain test
        print("\n  • In-Domain Test Set")
        test_id_results = self.trainer.evaluate(self.tokenized_datasets['test_id'])
        results['test_in_domain'] = self._format_results(test_id_results)
        self._print_metrics(results['test_in_domain'])
        
        # Get predictions for detailed analysis
        predictions_id = self.trainer.predict(self.tokenized_datasets['test_id'])
        results['test_in_domain']['predictions'] = predictions_id
        
        # OOD test
        print("\n  • Out-of-Domain Test Set")
        test_ood_results = self.trainer.evaluate(self.tokenized_datasets['test_ood'])
        results['test_ood'] = self._format_results(test_ood_results)
        self._print_metrics(results['test_ood'])
        
        # Get predictions for detailed analysis
        predictions_ood = self.trainer.predict(self.tokenized_datasets['test_ood'])
        results['test_ood']['predictions'] = predictions_ood
        
        # Save results
        self._save_results(results)
        
        # Generate detailed reports
        self._generate_detailed_reports(results)
        
        return results
    
    def _format_results(self, raw_results):
        """Format results dict"""
        formatted = {}
        for key, value in raw_results.items():
            if key.startswith('eval_'):
                formatted[key.replace('eval_', '')] = value
        return formatted
    
    def _print_metrics(self, metrics):
        """Print metrics in a nice format"""
        print(f"    Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"    Precision: {metrics.get('precision', 0):.4f}")
        print(f"    Recall:    {metrics.get('recall', 0):.4f}")
        print(f"    F1 Score:  {metrics.get('f1', 0):.4f}")
        if 'macro_f1' in metrics:
            print(f"    Macro F1:  {metrics.get('macro_f1', 0):.4f}")
    
    def _save_results(self, results):
        """Save results to JSON"""
        # Remove predictions from saved results (too large)
        results_to_save = {}
        for split, metrics in results.items():
            results_to_save[split] = {k: v for k, v in metrics.items() 
                                     if k != 'predictions'}
        
        results_file = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n  💾 Results saved to: {results_file}")
    
    def _generate_detailed_reports(self, results):
        """Generate classification reports and confusion matrices"""
        
        for split_name in ['test_in_domain', 'test_ood']:
            if 'predictions' not in results[split_name]:
                continue
            
            predictions = results[split_name]['predictions']
            y_true = predictions.label_ids
            y_pred = np.argmax(predictions.predictions, axis=-1)
            
            # Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=list(self.id2label.values()),
                digits=4,
                zero_division=0
            )
            
            report_file = os.path.join(
                self.output_dir,
                f'classification_report_{split_name}.txt'
            )
            with open(report_file, 'w') as f:
                f.write(f"Classification Report - {split_name}\n")
                f.write("="*70 + "\n\n")
                f.write(report)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_file = os.path.join(
                self.output_dir,
                f'confusion_matrix_{split_name}.csv'
            )
            cm_df = pd.DataFrame(
                cm,
                index=list(self.id2label.values()),
                columns=list(self.id2label.values())
            )
            cm_df.to_csv(cm_file)
            
            print(f"  📄 {split_name} report: {report_file}")
            print(f"  📄 {split_name} confusion matrix: {cm_file}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("TAGLISH GASLIGHTING DETECTION - MODEL TRAINING")
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
    
    # Train all models for both tasks
    for model_key in Config.MODELS.keys():
        for task in ['binary', 'tactic']:
            
            print("\n" + "="*70)
            print(f"MODEL: {Config.MODELS[model_key]['display']}")
            print(f"TASK: {task.upper()}")
            print("="*70)
            
            try:
                # Get datasets for this task
                datasets = data_loader.get_datasets(task=task)
                
                # Initialize trainer
                trainer = ModelTrainer(model_key, task, datasets)
                
                # Prepare data
                trainer.prepare_data()
                
                # Train
                train_result = trainer.train()
                
                # Evaluate
                eval_results = trainer.evaluate()
                
                # Store results
                all_results.append({
                    'model': Config.MODELS[model_key]['display'],
                    'model_key': model_key,
                    'task': task,
                    'val_accuracy': eval_results['validation']['accuracy'],
                    'val_f1': eval_results['validation']['f1'],
                    'test_id_accuracy': eval_results['test_in_domain']['accuracy'],
                    'test_id_f1': eval_results['test_in_domain']['f1'],
                    'test_ood_accuracy': eval_results['test_ood']['accuracy'],
                    'test_ood_f1': eval_results['test_ood']['f1'],
                    'status': '✅ Success'
                })
                
                # Cleanup
                del trainer
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n❌ Error training {model_key} ({task}): {str(e)}")
                all_results.append({
                    'model': Config.MODELS[model_key]['display'],
                    'model_key': model_key,
                    'task': task,
                    'val_accuracy': 0.0,
                    'val_f1': 0.0,
                    'test_id_accuracy': 0.0,
                    'test_id_f1': 0.0,
                    'test_ood_accuracy': 0.0,
                    'test_ood_f1': 0.0,
                    'status': f'❌ Failed: {str(e)[:50]}'
                })
                continue
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    print("\n" + results_df.to_string(index=False))
    
    # Save summary
    summary_file = os.path.join(Config.RESULTS_DIR, 'training_summary.csv')
    results_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Generate comparison tables for thesis
    generate_thesis_tables(results_df)
    
    print("\n" + "="*70)
    print("🎉 ALL TRAINING COMPLETED!")
    print("="*70)

def generate_thesis_tables(results_df):
    """Generate formatted tables for thesis"""
    
    print("\n" + "="*70)
    print("THESIS TABLES")
    print("="*70)
    
    # Separate binary and tactic results
    binary_results = results_df[results_df['task'] == 'binary'].copy()
    tactic_results = results_df[results_df['task'] == 'tactic'].copy()
    
    # Binary Classification Table
    print("\nTable 1: Binary Classification Results")
    print("-" * 70)
    binary_table = binary_results[['model', 'test_id_accuracy', 'test_id_f1', 
                                   'test_ood_accuracy', 'test_ood_f1']]
    binary_table.columns = ['Model', 'In-Domain Acc', 'In-Domain F1', 
                            'OOD Acc', 'OOD F1']
    print(binary_table.to_string(index=False))
    
    # Multi-class Table
    print("\nTable 2: Multi-class Tactic Classification Results")
    print("-" * 70)
    tactic_table = tactic_results[['model', 'test_id_accuracy', 'test_id_f1',
                                   'test_ood_accuracy', 'test_ood_f1']]
    tactic_table.columns = ['Model', 'In-Domain Acc', 'In-Domain F1',
                            'OOD Acc', 'OOD F1']
    print(tactic_table.to_string(index=False))
    
    # Save tables
    binary_table.to_csv(
        os.path.join(Config.RESULTS_DIR, 'thesis_table_binary.csv'),
        index=False
    )
    tactic_table.to_csv(
        os.path.join(Config.RESULTS_DIR, 'thesis_table_tactic.csv'),
        index=False
    )
    
    print(f"\n💾 Thesis tables saved to: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
