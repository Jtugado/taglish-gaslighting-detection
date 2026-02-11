import pandas as pd
import re
import json
import string
from datetime import datetime
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
# Cleaning thresholds (saved for reproducibility)
CONFIG = {
    "min_length": 15,
    "max_length": 600,
    "repeat_char_limit": 3,  # Increased from 2 to preserve paralinguistic cues
    "language_detection_threshold": 2,
    "version": "2.0_thesis_ready",
    "date": datetime.now().isoformat()
}

# Filipino markers for heuristic language detection
FILIPINO_MARKERS = ['ang', 'ng', 'sa', 'na', 'mo', 'ka', 'lang', 'yan', 'yung', 
                    'mga', 'din', 'rin', 'po', 'ba', 'naman', 'talaga']

# ================= CLEANING FUNCTIONS =================

def remove_urls(text):
    """Remove URLs from text"""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)
    return text

def remove_mentions(text):
    """Remove Reddit mentions and subreddit references"""
    text = re.sub(r'u/\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

def remove_special_characters(text):
    """
    Remove special characters while preserving:
    - Alphanumeric characters
    - Basic punctuation (.,!?-)
    - Filipino diacritics (áéíóúñ)
    
    Note: This is a simplified approach. Some Filipino characters (ü, Á, É, etc.)
    may be lost. This limitation is acceptable for our research scope as these
    characters are rare in social media text.
    """
    # Keep alphanumeric, spaces, and specified punctuation/diacritics
    text = re.sub(r'[^\w\s\.,!?\-áéíóúñÁÉÍÓÚÑ]', '', text, flags=re.IGNORECASE)
    return text

def normalize_repetition(text):
    """
    Limit repeated characters to preserve paralinguistic cues.
    
    Rationale: In gaslighting detection, elongation patterns like "hahahaha"
    can signal mockery or trivialization. We limit to 3 repetitions rather
    than 2 to preserve this signal while still reducing noise.
    
    Examples:
    - "hahahahaha" → "hahaha"
    - "nooooo" → "nooo"
    """
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)  # Changed from {2,} to {3,}
    return text

def normalize_whitespace(text):
    """Normalize excessive whitespace"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def clean_sentence(text):
    """
    Main cleaning pipeline for a sentence.
    
    Order matters:
    1. Remove URLs (before char cleaning to avoid partial matches)
    2. Remove mentions (before char cleaning)
    3. Remove special characters
    4. Normalize repetition (preserves some paralinguistic cues)
    5. Normalize whitespace (final cleanup)
    """
    if pd.isna(text) or text == "":
        return ""
    
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_special_characters(text)
    text = normalize_repetition(text)
    text = normalize_whitespace(text)
    
    return text.strip()

def filter_low_quality(df, stats_log):
    """
    Filter out low-quality samples with detailed logging.
    
    Criteria:
    - Length: 15-600 characters (empirically chosen after inspecting distributions)
    - Content: Not mostly numbers or punctuation
    - Uniqueness: No duplicate sentences
    
    Note on duplicates: Cleaning may make distinct sentences identical
    (e.g., "You are wrong!!!" → "You are wrong!"). This is acceptable
    as we prioritize semantic uniqueness over surface variation.
    """
    initial_count = len(df)
    stats_log['initial_samples'] = initial_count
    
    # Remove too short/long
    df = df[df['sentence'].str.len() >= CONFIG['min_length']]
    df = df[df['sentence'].str.len() <= CONFIG['max_length']]
    stats_log['after_length_filter'] = len(df)
    stats_log['dropped_by_length'] = initial_count - len(df)
    
    # Remove mostly numeric
    before_numeric = len(df)
    df = df[~df['sentence'].str.match(r'^[\d\s\.,]+$')]
    stats_log['after_numeric_filter'] = len(df)
    stats_log['dropped_by_numeric'] = before_numeric - len(df)
    
    # Remove mostly punctuation
    def is_mostly_punctuation(text):
        if len(text) == 0:
            return True
        punct_count = sum(1 for c in text if c in string.punctuation)
        return punct_count / len(text) > 0.5
    
    before_punct = len(df)
    df = df[~df['sentence'].apply(is_mostly_punctuation)]
    stats_log['after_punctuation_filter'] = len(df)
    stats_log['dropped_by_punctuation'] = before_punct - len(df)
    
    # Remove duplicates (after cleaning, so semantically similar sentences are deduplicated)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['sentence'], keep='first')
    stats_log['after_deduplication'] = len(df)
    stats_log['dropped_by_duplication'] = before_dedup - len(df)
    
    return df, stats_log

def validate_labels(df, stats_log):
    """
    Validate and fix label consistency.
    
    Checks:
    1. Binary labels are valid (gaslighting/non_gaslighting)
    2. Tactic labels are valid
    3. Consistency: non_gaslighting must have NA tactic
    """
    initial_count = len(df)
    
    # Check binary labels
    valid_binary = df['binary_label'].isin(['gaslighting', 'non_gaslighting'])
    invalid_binary_count = (~valid_binary).sum()
    if invalid_binary_count > 0:
        stats_log['invalid_binary_labels'] = int(invalid_binary_count)
        df = df[valid_binary]
    
    # Check tactic labels
    valid_tactics = ['distortion_denial', 'trivialization_minimization', 
                     'coercion_intimidation', 'knowledge_invalidation', 'NA']
    valid_tactic = df['tactic_label'].isin(valid_tactics)
    invalid_tactic_count = (~valid_tactic).sum()
    if invalid_tactic_count > 0:
        stats_log['invalid_tactic_labels'] = int(invalid_tactic_count)
        df = df[valid_tactic]
    
    # Check consistency: non-gaslighting should have NA tactic
    non_gas_mask = df['binary_label'] == 'non_gaslighting'
    inconsistent = non_gas_mask & (df['tactic_label'] != 'NA')
    inconsistent_count = inconsistent.sum()
    if inconsistent_count > 0:
        stats_log['fixed_inconsistent_labels'] = int(inconsistent_count)
        df.loc[inconsistent, 'tactic_label'] = 'NA'
    
    stats_log['valid_samples_after_label_check'] = len(df)
    return df, stats_log

def detect_language_heuristic(text):
    """
    Heuristic-based language detection for Filipino/mixed text.
    
    Method: Count presence of common Filipino function words.
    Threshold: 2+ markers → classified as Tagalog
    
    Limitations:
    - May fail for very short sentences
    - Mixed English-Tagalog may be undercounted
    - This is a heuristic approximation, not true language detection
    
    For thesis: Clearly state this is heuristic-based and acknowledge limitations.
    """
    text_lower = text.lower()
    text_words = set(text_lower.split())
    filipino_count = len(text_words.intersection(FILIPINO_MARKERS))
    return 'tagalog' if filipino_count >= CONFIG['language_detection_threshold'] else 'mixed'

def add_metadata(df):
    """
    Add metadata columns for analysis and error inspection.
    
    Metadata includes:
    - sentence_length: Character count
    - word_count: Token count
    - language_heuristic: Heuristic language classification (NOT true detection)
    """
    df['sentence_length'] = df['sentence'].str.len()
    df['word_count'] = df['sentence'].str.split().str.len()
    df['language_heuristic'] = df['sentence'].apply(detect_language_heuristic)
    
    return df

def balance_dataset(df, domain_type='in_domain'):
    """
    Balance dataset according to thesis requirements.
    
    For in-domain (political):
    - 250 of each gaslighting tactic
    - 1000 non-gaslighting
    
    For OOD (cultural/revisionist):
    - 37-38 of each gaslighting tactic per domain
    - 150 non-gaslighting per domain
    
    Strategy: If we have excess samples, randomly sample to target.
    If we have insufficient samples, keep all (will be noted in logs).
    """
    balanced_data = []
    balance_log = {}
    
    # Get unique domains
    domains = df['domain'].unique()
    
    for domain in domains:
        domain_df = df[df['domain'] == domain]
        balance_log[domain] = {}
        
        # Set targets based on domain type
        if domain == 'political':
            gaslighting_target = 250
            non_gaslighting_target = 1000
        else:  # cultural or revisionist
            gaslighting_target = 38
            non_gaslighting_target = 150
        
        # Balance gaslighting tactics
        for tactic in ['distortion_denial', 'trivialization_minimization', 
                       'coercion_intimidation', 'knowledge_invalidation']:
            tactic_df = domain_df[domain_df['tactic_label'] == tactic]
            actual_count = len(tactic_df)
            
            if actual_count > gaslighting_target:
                tactic_df = tactic_df.sample(n=gaslighting_target, random_state=42)
                balance_log[domain][tactic] = f"{gaslighting_target}/{actual_count} (sampled)"
            else:
                balance_log[domain][tactic] = f"{actual_count}/{gaslighting_target} (kept all)"
            
            balanced_data.append(tactic_df)
        
        # Balance non-gaslighting
        non_gas_df = domain_df[domain_df['binary_label'] == 'non_gaslighting']
        actual_count = len(non_gas_df)
        
        if actual_count > non_gaslighting_target:
            non_gas_df = non_gas_df.sample(n=non_gaslighting_target, random_state=42)
            balance_log[domain]['non_gaslighting'] = f"{non_gaslighting_target}/{actual_count} (sampled)"
        else:
            balance_log[domain]['non_gaslighting'] = f"{actual_count}/{non_gaslighting_target} (kept all)"
        
        balanced_data.append(non_gas_df)
    
    # Combine all balanced data
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    return balanced_df, balance_log

# ================= MAIN CLEANING PIPELINE =================

def clean_dataset(input_file, output_file, dataset_type='in_domain'):
    """
    Main cleaning pipeline with comprehensive logging.
    
    Args:
        input_file: Path to raw CSV file
        output_file: Path to save cleaned CSV
        dataset_type: 'in_domain' or 'ood'
    
    Returns:
        Cleaned DataFrame and statistics dictionary
    """
    print(f"\n{'='*70}")
    print(f"CLEANING {dataset_type.upper()} DATASET: {input_file}")
    print(f"{'='*70}")
    
    # Initialize logging
    stats_log = {
        'dataset_type': dataset_type,
        'input_file': input_file,
        'output_file': output_file,
        'config': CONFIG
    }
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} samples")
    
    # Initial statistics
    initial_stats = {
        'binary_label': df['binary_label'].value_counts().to_dict(),
        'tactic_label': df['tactic_label'].value_counts().to_dict(),
        'domain': df['domain'].value_counts().to_dict() if 'domain' in df.columns else {}
    }
    stats_log['initial_stats'] = initial_stats
    
    print("\n📊 Initial Statistics:")
    print(f"  Binary labels: {initial_stats['binary_label']}")
    print(f"  Tactic labels: {initial_stats['tactic_label']}")
    if initial_stats['domain']:
        print(f"  Domains: {initial_stats['domain']}")
    
    # Clean sentences
    print("\n🧹 Cleaning sentences...")
    df['sentence_original'] = df['sentence'].copy()
    df['sentence'] = df['sentence'].apply(clean_sentence)
    
    if 'context_window' in df.columns:
        df['context_window'] = df['context_window'].apply(clean_sentence)
    
    print("  ✅ Sentences cleaned")
    
    # Filter low quality
    print("\n🔍 Filtering low-quality samples...")
    df, stats_log = filter_low_quality(df, stats_log)
    print(f"  ✅ Quality filter complete")
    print(f"     Dropped by length: {stats_log['dropped_by_length']}")
    print(f"     Dropped by numeric: {stats_log['dropped_by_numeric']}")
    print(f"     Dropped by punctuation: {stats_log['dropped_by_punctuation']}")
    print(f"     Dropped by duplication: {stats_log['dropped_by_duplication']}")
    
    # Validate labels
    print("\n🏷️  Validating labels...")
    df, stats_log = validate_labels(df, stats_log)
    if 'fixed_inconsistent_labels' in stats_log:
        print(f"  ⚠️  Fixed {stats_log['fixed_inconsistent_labels']} inconsistent labels")
    print("  ✅ Labels validated")
    
    # Add metadata
    print("\n📝 Adding metadata...")
    df = add_metadata(df)
    print("  ✅ Metadata added (sentence_length, word_count, language_heuristic)")
    
    # Balance dataset (ACTUALLY CALLED NOW!)
    print("\n⚖️  Balancing dataset...")
    df, balance_log = balance_dataset(df, dataset_type)
    stats_log['balance_log'] = balance_log
    print("  ✅ Dataset balanced")
    for domain, tactics in balance_log.items():
        print(f"\n  {domain.upper()}:")
        for tactic, status in tactics.items():
            print(f"    {tactic}: {status}")
    
    # Final statistics
    final_stats = {
        'total_samples': len(df),
        'binary_label': df['binary_label'].value_counts().to_dict(),
        'tactic_label': df['tactic_label'].value_counts().to_dict(),
        'domain': df['domain'].value_counts().to_dict() if 'domain' in df.columns else {},
        'avg_sentence_length': float(df['sentence_length'].mean()),
        'avg_word_count': float(df['word_count'].mean()),
        'language_distribution': df['language_heuristic'].value_counts().to_dict()
    }
    stats_log['final_stats'] = final_stats
    
    print("\n📊 Final Statistics:")
    print(f"  Total samples: {final_stats['total_samples']}")
    print(f"  Binary labels: {final_stats['binary_label']}")
    print(f"  Tactic labels: {final_stats['tactic_label']}")
    if final_stats['domain']:
        print(f"  Domains: {final_stats['domain']}")
    print(f"  Avg sentence length: {final_stats['avg_sentence_length']:.1f} chars")
    print(f"  Avg word count: {final_stats['avg_word_count']:.1f} words")
    print(f"  Language heuristic: {final_stats['language_distribution']}")
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Save statistics log
    log_file = output_file.replace('.csv', '_stats.json')
    with open(log_file, 'w') as f:
        json.dump(stats_log, f, indent=2)
    print(f"  ✅ Saved data and statistics log ({log_file})")
    
    return df, stats_log

def create_training_splits(in_domain_file, ood_file):
    """
    Create train/val/test splits from cleaned datasets.
    
    Strategy:
    - In-domain split: 80/10/10 (train/val/test)
    - Stratification: Multi-class by tactic_label (includes NA for non-gaslighting)
    - OOD: Kept separate for zero-shot evaluation
    
    Note: We stratify on tactic_label which implicitly includes non-gaslighting (NA)
    as a class. This is multi-class stratification, not binary stratification.
    """
    print(f"\n{'='*70}")
    print("CREATING TRAINING SPLITS")
    print(f"{'='*70}")
    
    split_log = {}
    
    # Load in-domain data
    print("\n📂 Loading in-domain data...")
    df = pd.read_csv(in_domain_file)
    split_log['in_domain_total'] = len(df)
    print(f"  Loaded {len(df)} samples")
    
    # First split: 80% train, 20% temp (for val+test)
    # Using multi-class stratification on tactic_label
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['tactic_label']
    )
    
    # Second split: split temp into 50% val, 50% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['tactic_label']
    )
    
    split_log['train_count'] = len(train_df)
    split_log['val_count'] = len(val_df)
    split_log['test_count'] = len(test_df)
    
    print(f"\n📊 Split sizes (multi-class stratified by tactic):")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits with descriptive names
    train_df.to_csv('train_in_domain.csv', index=False)
    val_df.to_csv('val_in_domain.csv', index=False)
    test_df.to_csv('test_in_domain.csv', index=False)
    
    print("\n💾 Saved in-domain splits:")
    print("  - train_in_domain.csv")
    print("  - val_in_domain.csv")
    print("  - test_in_domain.csv")
    
    # Load and process OOD data
    print("\n📂 Processing OOD data...")
    ood_df = pd.read_csv(ood_file)
    split_log['ood_total'] = len(ood_df)
    print(f"  OOD samples: {len(ood_df)}")
    
    # Split OOD by domain
    cultural_df = ood_df[ood_df['domain'] == 'cultural']
    revisionist_df = ood_df[ood_df['domain'] == 'revisionist']
    
    split_log['ood_cultural_count'] = len(cultural_df)
    split_log['ood_revisionist_count'] = len(revisionist_df)
    
    cultural_df.to_csv('test_ood_cultural.csv', index=False)
    revisionist_df.to_csv('test_ood_revisionist.csv', index=False)
    
    print("\n💾 Saved OOD test sets:")
    print(f"  - test_ood_cultural.csv ({len(cultural_df)} samples)")
    print(f"  - test_ood_revisionist.csv ({len(revisionist_df)} samples)")
    
    # Create comprehensive summary
    summary = {
        'split': ['train', 'val', 'test_in_domain', 'test_ood_cultural', 
                  'test_ood_revisionist', 'total'],
        'samples': [
            len(train_df), 
            len(val_df), 
            len(test_df),
            len(cultural_df),
            len(revisionist_df),
            len(train_df) + len(val_df) + len(test_df) + len(cultural_df) + len(revisionist_df)
        ],
        'gaslighting': [
            len(train_df[train_df['binary_label']=='gaslighting']),
            len(val_df[val_df['binary_label']=='gaslighting']),
            len(test_df[test_df['binary_label']=='gaslighting']),
            len(cultural_df[cultural_df['binary_label']=='gaslighting']),
            len(revisionist_df[revisionist_df['binary_label']=='gaslighting']),
            len(df[df['binary_label']=='gaslighting']) + len(ood_df[ood_df['binary_label']=='gaslighting'])
        ],
        'non_gaslighting': [
            len(train_df[train_df['binary_label']=='non_gaslighting']),
            len(val_df[val_df['binary_label']=='non_gaslighting']),
            len(test_df[test_df['binary_label']=='non_gaslighting']),
            len(cultural_df[cultural_df['binary_label']=='non_gaslighting']),
            len(revisionist_df[revisionist_df['binary_label']=='non_gaslighting']),
            len(df[df['binary_label']=='non_gaslighting']) + len(ood_df[ood_df['binary_label']=='non_gaslighting'])
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('dataset_summary.csv', index=False)
    
    print("\n📊 Complete Dataset Summary:")
    print(summary_df.to_string(index=False))
    
    # Save split configuration
    split_log['summary'] = summary
    with open('split_config.json', 'w') as f:
        json.dump(split_log, f, indent=2)
    
    print("\n💾 Saved configuration:")
    print("  - dataset_summary.csv (human-readable)")
    print("  - split_config.json (machine-readable)")

# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    print("🚀 Starting Enhanced Data Cleaning Pipeline")
    print(f"Version: {CONFIG['version']}")
    print(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    print("="*70)
    
    # Save configuration for reproducibility
    with open('cleaning_config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print("\n💾 Saved cleaning_config.json for reproducibility")
    
    # Clean in-domain dataset
    in_domain_cleaned, in_domain_stats = clean_dataset(
        input_file='in_domain_political_dataset.csv',
        output_file='in_domain_political_cleaned.csv',
        dataset_type='in_domain'
    )
    
    # Clean OOD dataset
    ood_cleaned, ood_stats = clean_dataset(
        input_file='ood_zero_shot_dataset.csv',
        output_file='ood_zero_shot_cleaned.csv',
        dataset_type='ood'
    )
    
    # Create training splits
    create_training_splits(
        in_domain_file='in_domain_political_cleaned.csv',
        ood_file='ood_zero_shot_cleaned.csv'
    )
    
    print("\n" + "="*70)
    print("✅ DATA CLEANING COMPLETE!")
    print("="*70)
    
    print("\n📁 Generated Files:")
    print("\nCleaned Data:")
    print("  1. in_domain_political_cleaned.csv")
    print("  2. ood_zero_shot_cleaned.csv")
    
    print("\nTraining Splits:")
    print("  3. train_in_domain.csv (80% of in-domain)")
    print("  4. val_in_domain.csv (10% of in-domain)")
    print("  5. test_in_domain.csv (10% of in-domain)")
    
    print("\nEvaluation Sets (Zero-Shot):")
    print("  6. test_ood_cultural.csv")
    print("  7. test_ood_revisionist.csv")
    
    print("\nDocumentation:")
    print("  8. dataset_summary.csv (overview statistics)")
    print("  9. cleaning_config.json (reproducibility settings)")
    print("  10. split_config.json (split details)")
    print("  11. in_domain_political_cleaned_stats.json (detailed cleaning logs)")
    print("  12. ood_zero_shot_cleaned_stats.json (detailed cleaning logs)")
    
    print("\n📋 Key Improvements in This Version:")
    print("  ✓ Fixed: balance_dataset() now actually called")
    print("  ✓ Fixed: Removed unused stopwords import")
    print("  ✓ Improved: Repetition limit increased to 3 (preserves paralinguistic cues)")
    print("  ✓ Improved: Renamed 'language' to 'language_heuristic'")
    print("  ✓ Improved: Comprehensive logging with JSON stats files")
    print("  ✓ Improved: Clear file naming (train_in_domain.csv, test_ood_*.csv)")
    print("  ✓ Added: Configuration snapshot for reproducibility")
    print("  ✓ Added: Detailed documentation of limitations and methodology")
    
    print("\n🎓 Thesis Defense Ready:")
    print("  • All thresholds documented and justified")
    print("  • Label validation with automatic fixing")
    print("  • Multi-class stratification clearly stated")
    print("  • Heuristic limitations acknowledged")
    print("  • Complete audit trail with stats logs")
    
    print("\n📝 Next Steps:")
    print("  1. Review *_stats.json files for quality assurance")
    print("  2. Verify dataset_summary.csv matches thesis requirements")
    print("  3. Proceed to preprocessing (tokenization, encoding)")
    print("  4. Train model on train_in_domain.csv")
    print("  5. Validate on val_in_domain.csv")
    print("  6. Test on test_in_domain.csv")
    print("  7. Zero-shot evaluation on test_ood_*.csv")