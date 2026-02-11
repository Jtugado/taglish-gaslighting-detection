"""
Comprehensive Data Preprocessing Pipeline
For BERT, RoBERTa, and DistilBERT Training

Includes:
1. Data Cleaning (remove duplicates, handle missing values, fix encoding)
2. Data Integration (merge datasets, balance classes)
3. Data Transformation (text normalization, tokenization prep)
4. Data Reduction (remove low-quality samples, outlier detection)
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================

CONFIG = {
    # File paths
    'in_domain_path': '/mnt/user-data/uploads/in_domain_political_dataset_FINAL_with_aigeneration.csv',
    'ood_path': '/mnt/user-data/uploads/ood_zero_shot_dataset_witth_aigeneration.csv',
    
    # Output paths
    'train_output': 'training_dataset.csv',
    'val_output': 'validation_dataset.csv',
    'test_output': 'test_dataset.csv',
    'ood_output': 'ood_test_dataset.csv',
    
    # Data split ratios
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    
    # Quality thresholds
    'min_sentence_length': 10,  # characters
    'max_sentence_length': 512,  # characters (BERT limit)
    'min_confidence': 1,  # Keep all confidence levels initially
    
    # Balance settings
    'balance_classes': True,
    'balance_tactics': True,
}

# ================= STEP 1: DATA CLEANING =================

def clean_text(text):
    """
    Clean and normalize text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove Reddit-specific artifacts
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    
    # Remove excessive punctuation (keep 1-2 max)
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
    
    # Strip whitespace
    text = text.strip()
    
    return text


def remove_duplicates(df, text_column='sentence'):
    """
    Remove duplicate sentences while preserving higher confidence samples
    """
    print("\n🔍 Removing duplicates...")
    
    initial_count = len(df)
    
    # Sort by confidence (descending) so we keep higher confidence duplicates
    df = df.sort_values('confidence', ascending=False)
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=[text_column], keep='first')
    
    # Remove near-duplicates (same sentence, minor differences)
    df[text_column + '_cleaned'] = df[text_column].str.lower().str.strip()
    df = df.drop_duplicates(subset=[text_column + '_cleaned'], keep='first')
    df = df.drop(columns=[text_column + '_cleaned'])
    
    removed = initial_count - len(df)
    print(f"   Removed {removed} duplicates ({removed/initial_count*100:.1f}%)")
    
    return df.reset_index(drop=True)


def handle_missing_values(df):
    """
    Handle missing values in dataset
    """
    print("\n🔍 Handling missing values...")
    
    # Fill missing tactic_label with 'NA' for non-gaslighting
    df.loc[df['binary_label'] == 'non_gaslighting', 'tactic_label'] = 'NA'
    
    # Drop rows with missing critical fields
    critical_fields = ['sentence', 'binary_label']
    before = len(df)
    df = df.dropna(subset=critical_fields)
    after = len(df)
    
    if before != after:
        print(f"   Dropped {before - after} rows with missing critical fields")
    
    # Fill other missing values
    df['context_window'] = df['context_window'].fillna(df['sentence'])
    df['confidence'] = df['confidence'].fillna(2)  # Default to medium confidence
    df['rationale'] = df['rationale'].fillna('No rationale provided')
    
    return df


def filter_quality(df, min_length=10, max_length=512):
    """
    Filter out low-quality samples
    """
    print("\n🔍 Filtering by quality...")
    
    initial_count = len(df)
    
    # Calculate sentence lengths
    df['sentence_length'] = df['sentence'].str.len()
    
    # Filter by length
    df = df[
        (df['sentence_length'] >= min_length) & 
        (df['sentence_length'] <= max_length)
    ]
    
    # Remove sentences that are just punctuation or numbers
    df = df[~df['sentence'].str.match(r'^[\W\d\s]+$')]
    
    # Remove sentences with excessive special characters (>50% of content)
    df['special_char_ratio'] = df['sentence'].apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in str(x)) / max(len(str(x)), 1)
    )
    df = df[df['special_char_ratio'] < 0.5]
    df = df.drop(columns=['special_char_ratio'])
    
    removed = initial_count - len(df)
    print(f"   Removed {removed} low-quality samples ({removed/initial_count*100:.1f}%)")
    
    return df.drop(columns=['sentence_length']).reset_index(drop=True)


def clean_dataset(df, dataset_name="Dataset"):
    """
    Complete cleaning pipeline
    """
    print("=" * 80)
    print(f"CLEANING: {dataset_name}")
    print("=" * 80)
    print(f"Initial size: {len(df)} samples")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['sentence'] = df['sentence'].apply(clean_text)
    df['context_window'] = df['context_window'].apply(clean_text)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Filter quality
    df = filter_quality(
        df, 
        min_length=CONFIG['min_sentence_length'],
        max_length=CONFIG['max_sentence_length']
    )
    
    print(f"\n✅ Final size: {len(df)} samples")
    print(f"   Retention rate: {len(df)/len(df)*100:.1f}%")
    
    return df


# ================= STEP 2: DATA INTEGRATION =================

def balance_classes(df, target_column='binary_label'):
    """
    Balance binary classes (gaslighting vs non-gaslighting)
    """
    print("\n⚖️ Balancing binary classes...")
    
    class_counts = df[target_column].value_counts()
    print(f"   Before: {dict(class_counts)}")
    
    # Find minority class size
    min_class_size = class_counts.min()
    
    # Undersample majority class
    balanced_dfs = []
    for label in df[target_column].unique():
        class_df = df[df[target_column] == label]
        if len(class_df) > min_class_size:
            # Prioritize higher confidence samples when undersampling
            class_df = class_df.sort_values('confidence', ascending=False)
            class_df = class_df.head(min_class_size)
        balanced_dfs.append(class_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    class_counts_after = df_balanced[target_column].value_counts()
    print(f"   After: {dict(class_counts_after)}")
    
    return df_balanced


def balance_tactics(df):
    """
    Balance gaslighting tactics
    """
    print("\n⚖️ Balancing gaslighting tactics...")
    
    # Only balance gaslighting samples
    gaslighting_df = df[df['binary_label'] == 'gaslighting'].copy()
    non_gaslighting_df = df[df['binary_label'] == 'non_gaslighting'].copy()
    
    if len(gaslighting_df) == 0:
        print("   No gaslighting samples to balance")
        return df
    
    tactic_counts = gaslighting_df['tactic_label'].value_counts()
    print(f"   Before: {dict(tactic_counts)}")
    
    # Find minimum tactic size
    min_tactic_size = tactic_counts.min()
    
    # Undersample each tactic
    balanced_tactics = []
    for tactic in gaslighting_df['tactic_label'].unique():
        tactic_df = gaslighting_df[gaslighting_df['tactic_label'] == tactic]
        if len(tactic_df) > min_tactic_size:
            tactic_df = tactic_df.sort_values('confidence', ascending=False)
            tactic_df = tactic_df.head(min_tactic_size)
        balanced_tactics.append(tactic_df)
    
    gaslighting_balanced = pd.concat(balanced_tactics, ignore_index=True)
    
    tactic_counts_after = gaslighting_balanced['tactic_label'].value_counts()
    print(f"   After: {dict(tactic_counts_after)}")
    
    # Combine with non-gaslighting
    df_balanced = pd.concat([gaslighting_balanced, non_gaslighting_df], ignore_index=True)
    
    return df_balanced


def integrate_datasets(in_domain_df, ood_df):
    """
    Integrate in-domain and OOD datasets
    """
    print("\n" + "=" * 80)
    print("DATA INTEGRATION")
    print("=" * 80)
    
    print(f"\nIn-domain dataset: {len(in_domain_df)} samples")
    print(f"OOD dataset: {len(ood_df)} samples")
    
    # Rename 'macro_domain' to 'domain' in OOD dataset for consistency
    if 'macro_domain' in ood_df.columns:
        ood_df = ood_df.rename(columns={'macro_domain': 'domain'})
    
    # Ensure same columns
    common_columns = list(set(in_domain_df.columns) & set(ood_df.columns))
    in_domain_df = in_domain_df[common_columns]
    ood_df = ood_df[common_columns]
    
    # Add dataset source indicator
    in_domain_df['dataset_source'] = 'in_domain'
    ood_df['dataset_source'] = 'ood'
    
    # Combined training dataset (in-domain only)
    training_df = in_domain_df.copy()
    
    # OOD stays separate for zero-shot evaluation
    ood_test_df = ood_df.copy()
    
    print(f"\n✅ Training dataset: {len(training_df)} samples")
    print(f"✅ OOD test dataset: {len(ood_test_df)} samples")
    
    return training_df, ood_test_df


# ================= STEP 3: DATA TRANSFORMATION =================

def normalize_labels(df):
    """
    Normalize and encode labels
    """
    print("\n🔧 Normalizing labels...")
    
    # Ensure consistent label format
    df['binary_label'] = df['binary_label'].str.lower().str.strip()
    df['tactic_label'] = df['tactic_label'].str.lower().str.strip()
    
    # Create numeric labels for binary classification
    label_map = {'gaslighting': 1, 'non_gaslighting': 0}
    df['label'] = df['binary_label'].map(label_map)
    
    # Create numeric labels for multi-class (tactic) classification
    tactics = df[df['binary_label'] == 'gaslighting']['tactic_label'].unique()
    tactic_map = {tactic: i+1 for i, tactic in enumerate(sorted(tactics))}
    tactic_map['na'] = 0  # Non-gaslighting gets 0
    
    df['tactic_label_encoded'] = df['tactic_label'].map(tactic_map)
    
    # Fill any missing encodings
    df['tactic_label_encoded'] = df['tactic_label_encoded'].fillna(0).astype(int)
    
    print(f"   Binary labels: {label_map}")
    print(f"   Tactic labels: {tactic_map}")
    
    return df, label_map, tactic_map


def add_features(df):
    """
    Add useful features for model training
    """
    print("\n🔧 Adding features...")
    
    # Text length features
    df['text_length'] = df['sentence'].str.len()
    df['word_count'] = df['sentence'].str.split().str.len()
    
    # Context availability
    df['has_context'] = (df['context_window'] != df['sentence']).astype(int)
    
    # Confidence as categorical
    df['confidence_category'] = pd.cut(
        df['confidence'], 
        bins=[0, 1, 2, 3], 
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    
    print(f"   Added: text_length, word_count, has_context, confidence_category")
    
    return df


# ================= STEP 4: DATA REDUCTION & SPLITTING =================

def remove_outliers(df):
    """
    Remove statistical outliers
    """
    print("\n🔍 Removing outliers...")
    
    initial_count = len(df)
    
    # Remove sentences that are too short or too long (statistical outliers)
    q1 = df['text_length'].quantile(0.01)
    q99 = df['text_length'].quantile(0.99)
    
    df = df[(df['text_length'] >= q1) & (df['text_length'] <= q99)]
    
    removed = initial_count - len(df)
    print(f"   Removed {removed} outliers ({removed/initial_count*100:.1f}%)")
    
    return df


def stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Stratified split maintaining class distribution
    """
    print("\n📊 Creating stratified splits...")
    
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df['binary_label'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_adjusted),
        stratify=temp_df['binary_label'],
        random_state=random_state
    )
    
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print("\n   Label distribution:")
    print(f"   Train: {dict(train_df['binary_label'].value_counts())}")
    print(f"   Val: {dict(val_df['binary_label'].value_counts())}")
    print(f"   Test: {dict(test_df['binary_label'].value_counts())}")
    
    return train_df, val_df, test_df


# ================= MAIN PIPELINE =================

def preprocess_pipeline():
    """
    Complete preprocessing pipeline
    """
    
    print("=" * 80)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        if not key.endswith('_path') and not key.endswith('_output'):
            print(f"   {key}: {value}")
    
    # ===== STEP 1: LOAD DATA =====
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    in_domain_df = pd.read_csv(CONFIG['in_domain_path'])
    ood_df = pd.read_csv(CONFIG['ood_path'])
    
    print(f"✅ Loaded in-domain: {len(in_domain_df)} samples")
    print(f"✅ Loaded OOD: {len(ood_df)} samples")
    
    # ===== STEP 2: CLEAN DATA =====
    print("\n" + "=" * 80)
    print("STEP 2: DATA CLEANING")
    print("=" * 80)
    
    in_domain_df = clean_dataset(in_domain_df, "In-Domain Dataset")
    ood_df = clean_dataset(ood_df, "OOD Dataset")
    
    # ===== STEP 3: INTEGRATE DATA =====
    training_df, ood_test_df = integrate_datasets(in_domain_df, ood_df)
    
    # ===== STEP 4: BALANCE DATA =====
    if CONFIG['balance_classes']:
        print("\n" + "=" * 80)
        print("STEP 4: DATA BALANCING")
        print("=" * 80)
        
        training_df = balance_classes(training_df)
        
        if CONFIG['balance_tactics']:
            training_df = balance_tactics(training_df)
    
    # ===== STEP 5: TRANSFORM DATA =====
    print("\n" + "=" * 80)
    print("STEP 5: DATA TRANSFORMATION")
    print("=" * 80)
    
    training_df, label_map, tactic_map = normalize_labels(training_df)
    ood_test_df, _, _ = normalize_labels(ood_test_df)
    
    training_df = add_features(training_df)
    ood_test_df = add_features(ood_test_df)
    
    # ===== STEP 6: REDUCE & SPLIT DATA =====
    print("\n" + "=" * 80)
    print("STEP 6: DATA REDUCTION & SPLITTING")
    print("=" * 80)
    
    training_df = remove_outliers(training_df)
    
    train_df, val_df, test_df = stratified_split(
        training_df,
        train_ratio=CONFIG['train_ratio'],
        val_ratio=CONFIG['val_ratio'],
        test_ratio=CONFIG['test_ratio']
    )
    
    # ===== STEP 7: SAVE PROCESSED DATA =====
    print("\n" + "=" * 80)
    print("STEP 7: SAVING PROCESSED DATA")
    print("=" * 80)
    
    # Save datasets
    train_df.to_csv(CONFIG['train_output'], index=False)
    val_df.to_csv(CONFIG['val_output'], index=False)
    test_df.to_csv(CONFIG['test_output'], index=False)
    ood_test_df.to_csv(CONFIG['ood_output'], index=False)
    
    print(f"\n✅ Saved training set: {CONFIG['train_output']} ({len(train_df)} samples)")
    print(f"✅ Saved validation set: {CONFIG['val_output']} ({len(val_df)} samples)")
    print(f"✅ Saved test set: {CONFIG['test_output']} ({len(test_df)} samples)")
    print(f"✅ Saved OOD test set: {CONFIG['ood_output']} ({len(ood_test_df)} samples)")
    
    # ===== STEP 8: GENERATE STATISTICS =====
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    print("\n📊 Training Set:")
    print(f"   Total samples: {len(train_df)}")
    print(f"   Gaslighting: {(train_df['label']==1).sum()}")
    print(f"   Non-gaslighting: {(train_df['label']==0).sum()}")
    print(f"   Avg sentence length: {train_df['text_length'].mean():.1f} chars")
    print(f"   Avg word count: {train_df['word_count'].mean():.1f} words")
    
    print("\n📊 Tactic Distribution (Training):")
    tactic_dist = train_df[train_df['binary_label'] == 'gaslighting']['tactic_label'].value_counts()
    for tactic, count in tactic_dist.items():
        print(f"   {tactic}: {count}")
    
    print("\n📊 OOD Test Set:")
    print(f"   Total samples: {len(ood_test_df)}")
    print(f"   Domains: {dict(ood_test_df['domain'].value_counts())}")
    print(f"   Gaslighting: {(ood_test_df['label']==1).sum()}")
    print(f"   Non-gaslighting: {(ood_test_df['label']==0).sum()}")
    
    # Save label mappings
    label_info = {
        'binary_label_map': label_map,
        'tactic_label_map': tactic_map
    }
    
    import json
    with open('label_mappings.json', 'w') as f:
        json.dump(label_info, f, indent=2)
    
    print(f"\n✅ Saved label mappings: label_mappings.json")
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE - DATASETS READY FOR TRAINING")
    print("=" * 80)
    
    return train_df, val_df, test_df, ood_test_df


if __name__ == "__main__":
    train_df, val_df, test_df, ood_test_df = preprocess_pipeline()
