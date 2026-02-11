import pandas as pd
import re
import os

# ================= CONFIGURATION =================
FILE_TRAIN = "in_domain_political_dataset_FINAL_with_aigeneration.csv"
FILE_OOD = "ood_zero_shot_dataset_witth_aigeneration.csv"

# Thesis Targets
TARGET_TRAIN_TOTAL = 2000
TARGET_OOD_TOTAL = 600
TARGET_TACTIC_BALANCE = 250  # per tactic in Train

# False Positive Patterns (For Final Audit)
PATTERNS_TO_FLAG = {
    "Reporter_Perspective": [
        r"they (say|said|call|called).*bobo",
        r"they (say|said|call|called).*tanga",
        r"sinabi.*bobo",
        r"sinabi.*tanga",
        r"tinawag.*bobo",
    ],
    "Meta_Discussion": [
        r"can i call",
        r"pwede ba",
        r"allowed ba",
        r"is this gaslighting"
    ],
    "Wrong_Context": [
        r"makikita mo sa news",
        r"makikita mo na",
        r"move on to the next topic",
        r"chill crowd",
        r"chill lang"
    ]
}

def check_false_positives(df, filename):
    print(f"\n🔍 Scanning {filename} for potential False Positives...")
    flagged_count = 0
    
    for idx, row in df.iterrows():
        if row['binary_label'] != 'gaslighting':
            continue
            
        sentence = str(row['sentence']).lower()
        context = str(row.get('context_window_ref', '')).lower()
        text_blob = f"{sentence} {context}"
        
        issues = []
        
        # Check Patterns
        for issue_name, patterns in PATTERNS_TO_FLAG.items():
            for pat in patterns:
                if re.search(pat, text_blob):
                    issues.append(issue_name)
                    break
        
        if issues:
            flagged_count += 1
            if flagged_count <= 5: # Show first 5 examples
                print(f"   ⚠️ Row {idx} Flagged [{', '.join(issues)}]: \"{row['sentence'][:60]}...\"")

    if flagged_count == 0:
        print("   ✅ Clean! No obvious false positive patterns found.")
    else:
        print(f"   ⚠️ Found {flagged_count} potential false positives. Please verify them manually.")

def validate_datasets():
    print("="*60)
    print("🎓 THESIS DATASET VALIDATION REPORT")
    print("="*60)

    # 1. LOAD FILES
    if not os.path.exists(FILE_TRAIN) or not os.path.exists(FILE_OOD):
        print("❌ Error: Files not found!")
        print(f"   Looking for: {FILE_TRAIN} and {FILE_OOD}")
        return

    df_train = pd.read_csv(FILE_TRAIN)
    df_ood = pd.read_csv(FILE_OOD)

    print(f"📂 Loaded Train Set: {len(df_train)} rows")
    print(f"📂 Loaded OOD Set:   {len(df_ood)} rows")

    # 2. CHECK DATA LEAKAGE (CRITICAL)
    print("\n🔒 Checking for Data Leakage (Train vs OOD)...")
    train_ids = set(df_train['id'].astype(str))
    ood_ids = set(df_ood['id'].astype(str))
    
    overlap = train_ids.intersection(ood_ids)
    
    if len(overlap) > 0:
        print(f"   ❌ CRITICAL FAIL: Found {len(overlap)} duplicate rows between Train and OOD!")
        print("   Action Required: Remove these IDs from OOD set immediately.")
    else:
        print("   ✅ PASS: Zero overlap. Datasets are strictly isolated.")

    # 3. TRAIN SET AUDIT
    print(f"\n📊 [TRAIN SET] Distribution Check ({FILE_TRAIN})")
    
    # Binary Split
    binary_counts = df_train['binary_label'].value_counts()
    print(f"   Binary Split:")
    print(f"     - Non-Gaslighting: {binary_counts.get('non_gaslighting', 0)} (Target: ~1000)")
    print(f"     - Gaslighting:     {binary_counts.get('gaslighting', 0)} (Target: ~1000)")
    
    # Tactic Balance
    print(f"   Tactic Balance (Target: ~{TARGET_TACTIC_BALANCE} each):")
    tactic_counts = df_train[df_train['binary_label']=='gaslighting']['tactic_label'].value_counts()
    for tactic, count in tactic_counts.items():
        status = "✅" if 220 <= count <= 280 else "⚠️ Imbalanced"
        print(f"     - {tactic}: {count} {status}")

    # False Positive Scan
    check_false_positives(df_train, "TRAIN SET")

    # 4. OOD SET AUDIT
    print(f"\n📊 [OOD SET] Distribution Check ({FILE_OOD})")
    print(f"   Total Size: {len(df_ood)} (Target: ~600)")
    
    ood_domains = df_ood['macro_domain'].value_counts()
    print(f"   Domain Split:")
    for domain, count in ood_domains.items():
        print(f"     - {domain}: {count}")

    # False Positive Scan
    check_false_positives(df_ood, "OOD SET")

    # 5. FINAL VERDICT
    print("\n" + "="*60)
    print("📢 FINAL VERDICT")
    print("="*60)
    
    ready = True
    if len(overlap) > 0: ready = False
    if len(df_train) < 1800: 
        print("   ⚠️ Warning: Train set is smaller than 2000.")
    if len(df_ood) < 500:
        print("   ⚠️ Warning: OOD set is smaller than 600.")
        
    if ready:
        print("✅ DATASETS ARE STRUCTURED CORRECTLY FOR THESIS TRAINING.")
        print("   Proceed to 'train_super_model.py' if the 'Imbalanced' warnings are minor.")
    else:
        print("❌ DATASETS NEED FIXING BEFORE TRAINING.")

if __name__ == "__main__":
    validate_datasets()