# cleanup_github_folder.py

import os
import shutil
import glob
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

class CleanupConfig:
    """Configuration for cleanup"""
    
    # Base directory (current location)
    BASE_DIR = r"C:\Users\TokyosaurusRex\Desktop\Thesis\Taglish_Gaslighting_V2_github"
    
    # Files to KEEP (essential for GitHub)
    ESSENTIAL_FILES = [
        # Training scripts
        "taglish_gaslighting_training_final.py",
        "train_model_v2.py",  # Keep if exists
        
        # Test app
        "test_app.py",
        "test_app_fixed.py",
        
        # Data files - FINAL versions only
        "training_dataset.csv",
        "validation_dataset.csv", 
        "test_dataset.csv",
        "ood_test_dataset.csv",
        
        # Requirements
        "requirements.txt",
        "requirements_training.txt",
        
        # Documentation (will create if missing)
        "README.md",
        "MODEL_DOWNLOAD.md",
        "TEST_APP_GUIDE.md",
        
        # Data utilities
        "data_preprocessing_pipeline.py",
        "validate_thesis_datasets.py",
        "data_cleaning.py",
        
        # Label mapping
        "label_mappings.json"
    ]
    
    # Folders to KEEP
    ESSENTIAL_FOLDERS = [
        "results",           # Evaluation results
        "plots",             # Visualizations (if exists)
        "error_analysis",    # Error analysis (if exists)
        "evaluation_results" # Evaluation metrics (if exists)
    ]
    
    # Files/Folders to DELETE
    DELETE_PATTERNS = [
        "__pycache__",
        "*.pyc",
        ".ipynb_checkpoints",
        "venv",
        ".gradio",
        
        # Intermediate/duplicate datasets
        "*_cleaned.csv",
        "*_cleaned_revi*.csv",
        "*_with_aigeneration.csv",
        "corrected_review_cases.csv",
        "manual_review_needed",
        
        # Batch test files
        "batch_test_input.csv",
        "batch_predictions.csv",
        
        # Redundant scripts
        "generate_thesis_splits.py",
        "reddit_scraper_enhanced_v2.py",
        "correct_and_merge.py",
        
        # Old/versioned files (keep only latest)
        "in_domain_political_dataset_cleaned.csv",  # Superseded by training_dataset
        "in_domain_political_dataset_FINAL_with_*.csv",  # Superseded
        "in_domain_political_dataset_FINAL.csv",  # Superseded
        "ood_zero_shot_dataset_witth_aigeneration.csv",  # Typo version
    ]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_folder_size(folder_path):
    """Get total size of folder in MB"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total += os.path.getsize(fp)
    except Exception as e:
        print(f"Error calculating size: {e}")
    return total / (1024 * 1024)  # Convert to MB

def safe_delete(path):
    """Safely delete file or folder"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return True
    except Exception as e:
        print(f"  ⚠️ Could not delete {path}: {e}")
        return False
    return False

# ==========================================
# CLEANUP FUNCTIONS
# ==========================================

def backup_current_state():
    """Create a backup before cleanup"""
    print("\n📦 Creating backup...")
    
    backup_dir = CleanupConfig.BASE_DIR + "_BACKUP"
    
    if os.path.exists(backup_dir):
        print(f"  ⚠️ Backup already exists at: {backup_dir}")
        response = input("  Overwrite existing backup? (y/n): ").strip().lower()
        if response != 'y':
            print("  Backup skipped")
            return False
        shutil.rmtree(backup_dir)
    
    try:
        shutil.copytree(CleanupConfig.BASE_DIR, backup_dir)
        print(f"  ✓ Backup created: {backup_dir}")
        return True
    except Exception as e:
        print(f"  ❌ Backup failed: {e}")
        return False

def delete_unnecessary_files():
    """Delete files that shouldn't be in GitHub repo"""
    print("\n🗑️ Removing unnecessary files...")
    
    deleted_count = 0
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    # Delete by pattern
    for pattern in CleanupConfig.DELETE_PATTERNS:
        matches = glob.glob(pattern, recursive=False)
        for match in matches:
            print(f"  Deleting: {match}")
            if safe_delete(match):
                deleted_count += 1
    
    print(f"\n  ✓ Deleted {deleted_count} items")

def organize_results():
    """Organize result files into proper folders"""
    print("\n📁 Organizing results...")
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    # Create folders if they don't exist
    folders = {
        'evaluation_results': [],
        'error_analysis': [],
        'plots': []
    }
    
    for folder in folders.keys():
        os.makedirs(folder, exist_ok=True)
    
    # Check if results folder has evaluation data
    if os.path.exists('results'):
        print("  Moving files from results/ to evaluation_results/")
        for item in os.listdir('results'):
            src = os.path.join('results', item)
            dst = os.path.join('evaluation_results', item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print(f"    Moved: {item}")
        
        # Remove empty results folder
        try:
            if not os.listdir('results'):
                os.rmdir('results')
                print("  ✓ Removed empty results/ folder")
        except:
            pass
    
    print("  ✓ Organization complete")

def create_essential_docs():
    """Create essential documentation files if missing"""
    print("\n📝 Creating documentation...")
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    # .gitignore
    if not os.path.exists('.gitignore'):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
.ipynb_checkpoints

# Model files (too large for GitHub)
*.bin
*.safetensors
*.pth
*.pt
model_outputs/*/pytorch_model.bin
model_outputs/*/model.safetensors

# Large data
*.zip
*.tar.gz

# Gradio
.gradio/

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary
*.tmp
*.bak
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("  ✓ Created .gitignore")
    
    # README.md (simple version if doesn't exist)
    if not os.path.exists('README.md'):
        readme_content = """# Taglish Gaslighting Detection System

Detecting manipulation tactics in Taglish (Tagalog-English) political discourse.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models (see MODEL_DOWNLOAD.md)

3. Run test app:
   ```bash
   python test_app_fixed.py
   ```

## Repository Structure

- `taglish_gaslighting_training_final.py` - Training script
- `test_app_fixed.py` - Interactive testing interface
- `training_dataset.csv` - Training data
- `evaluation_results/` - Model performance metrics
- `model_outputs/` - Trained models (download separately)

## Performance

See `evaluation_results/` for complete metrics.

## Citation

[Add your citation information]

## License

[Add your license]
"""
        with open('README.md', 'w') as f:
            f.write(readme_content)
        print("  ✓ Created README.md")
    
    # MODEL_DOWNLOAD.md
    if not os.path.exists('MODEL_DOWNLOAD.md'):
        model_download = """# Model Download Guide

## Models Not Included

Model files (*.bin, *.safetensors) are not included in this repository due to GitHub's file size limits.

## Download Options

### Option 1: Hugging Face Hub
[Add your Hugging Face links when uploaded]

### Option 2: Google Drive
[Add your Google Drive link]

### Option 3: Train From Scratch
```bash
python taglish_gaslighting_training_final.py
```

## Expected Structure

After downloading, your `model_outputs/` should contain:
- roberta-tagalog_binary/
- roberta-tagalog_tactic/
- mbert_binary/
- mbert_tactic/
- xlm-roberta_binary/
- xlm-roberta_tactic/
"""
        with open('MODEL_DOWNLOAD.md', 'w') as f:
            f.write(model_download)
        print("  ✓ Created MODEL_DOWNLOAD.md")
    
    # requirements.txt (if doesn't exist)
    if not os.path.exists('requirements.txt'):
        requirements = """torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
gradio>=4.0.0
tqdm>=4.65.0
"""
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("  ✓ Created requirements.txt")

def check_file_sizes():
    """Check for files that might be too large for GitHub"""
    print("\n📊 Checking file sizes...")
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    large_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip model_outputs folder
        if 'model_outputs' in root:
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                size_mb = size / (1024 * 1024)
                
                if size_mb > 100:  # GitHub limit
                    large_files.append((filepath, size_mb))
                elif size_mb > 50:  # Warning threshold
                    print(f"  ⚠️ Large file: {filepath} ({size_mb:.1f} MB)")
            except:
                pass
    
    if large_files:
        print("\n  ❌ Files exceeding GitHub's 100MB limit:")
        for filepath, size in large_files:
            print(f"    {filepath}: {size:.1f} MB")
        print("\n  These files should be:")
        print("    1. Added to .gitignore")
        print("    2. Uploaded separately (Google Drive, etc.)")
    else:
        print("  ✓ All files are within GitHub limits")
    
    return large_files

def create_folder_structure_doc():
    """Document the folder structure"""
    print("\n📋 Creating STRUCTURE.md...")
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    structure = """# Repository Structure

```
taglish-gaslighting-detection/
│
├── README.md                                    # Main documentation
├── MODEL_DOWNLOAD.md                            # How to get models
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
│
├── taglish_gaslighting_training_final.py        # Main training script
├── test_app_fixed.py                            # Gradio testing interface
├── data_preprocessing_pipeline.py               # Data preparation
├── validate_thesis_datasets.py                  # Dataset validation
│
├── training_dataset.csv                         # Training data
├── validation_dataset.csv                       # Validation data
├── test_dataset.csv                             # In-domain test
├── ood_test_dataset.csv                         # Out-of-domain test
│
├── model_outputs/                               # Trained models (download separately)
│   ├── roberta-tagalog_binary/
│   │   ├── config.json
│   │   ├── tokenizer files
│   │   └── pytorch_model.bin (not in repo)
│   ├── roberta-tagalog_tactic/
│   ├── mbert_binary/
│   ├── mbert_tactic/
│   ├── xlm-roberta_binary/
│   └── xlm-roberta_tactic/
│
├── evaluation_results/                          # Model performance metrics
│   ├── comprehensive_training_summary.csv
│   ├── thesis_table1_binary_with_metrics.csv
│   └── thesis_table2_tactic_with_metrics.csv
│
├── error_analysis/                              # Misclassification analysis
│   └── [model]_[task]_errors_[split].csv
│
└── plots/                                       # Visualizations
    ├── confusion_matrices/
    └── roc_curves/
```

## File Descriptions

### Core Scripts
- `taglish_gaslighting_training_final.py` - Trains all models with comprehensive evaluation
- `test_app_fixed.py` - Interactive Gradio app for testing models
- `data_preprocessing_pipeline.py` - Prepares and cleans datasets

### Data Files
- `training_dataset.csv` - Training samples (political domain)
- `validation_dataset.csv` - Validation samples
- `test_dataset.csv` - In-domain test (political)
- `ood_test_dataset.csv` - Out-of-domain test (cultural & revisionist)

### Results
- `evaluation_results/` - Complete performance metrics
- `error_analysis/` - Detailed error breakdowns for manual analysis
- `plots/` - Confusion matrices and ROC curves

### Models
- `model_outputs/` - Contains 6 trained models (3 models × 2 tasks)
- Model weights (*.bin files) must be downloaded separately
- Config files and tokenizers are included
"""
    
    with open('STRUCTURE.md', 'w') as f:
        f.write(structure)
    
    print("  ✓ Created STRUCTURE.md")

def generate_summary():
    """Generate final summary report"""
    print("\n" + "="*70)
    print("📊 CLEANUP SUMMARY")
    print("="*70)
    
    os.chdir(CleanupConfig.BASE_DIR)
    
    # Count files
    total_files = sum([len(files) for r, d, files in os.walk('.')])
    
    # Calculate total size
    total_size = get_folder_size('.')
    
    # Check for essential files
    print("\n✅ Essential Files:")
    for file in CleanupConfig.ESSENTIAL_FILES:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️ {file} (missing)")
    
    print(f"\n📊 Repository Statistics:")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size:.2f} MB")
    
    if total_size > 1000:
        print(f"  ⚠️ Size is large. Consider removing more files.")
    elif total_size > 500:
        print(f"  ⚠️ Size is moderate. Check for unnecessary data files.")
    else:
        print(f"  ✓ Size is good for GitHub")
    
    print("\n📁 Folder Structure:")
    for item in sorted(os.listdir('.')):
        if os.path.isdir(item) and not item.startswith('.'):
            size = get_folder_size(item)
            print(f"  📁 {item}/ ({size:.1f} MB)")
    
    print("\n" + "="*70)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("="*70)
    print("GITHUB FOLDER CLEANUP")
    print("="*70)
    print(f"\nWorking directory: {CleanupConfig.BASE_DIR}")
    print("\nThis script will:")
    print("  1. Create a backup")
    print("  2. Remove unnecessary files")
    print("  3. Organize results")
    print("  4. Create documentation")
    print("  5. Check file sizes")
    print("\n⚠️ WARNING: This will delete files!")
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n❌ Cleanup cancelled")
        return
    
    # Create backup
    if not backup_current_state():
        response = input("\nBackup failed. Continue anyway? (yes/no): ").strip().lower()
        if response != 'yes':
            print("\n❌ Cleanup cancelled")
            return
    
    # Run cleanup steps
    delete_unnecessary_files()
    organize_results()
    create_essential_docs()
    create_folder_structure_doc()
    
    # Check for issues
    large_files = check_file_sizes()
    
    # Generate summary
    generate_summary()
    
    # Final instructions
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE!")
    print("="*70)
    
    print("\n📝 Next Steps:")
    print("\n1. Review the changes:")
    print(f"   - Backup is at: {CleanupConfig.BASE_DIR}_BACKUP")
    print(f"   - Check STRUCTURE.md for organization")
    
    print("\n2. Initialize Git repository:")
    print(f"   cd \"{CleanupConfig.BASE_DIR}\"")
    print("   git init")
    print("   git add .")
    print("   git commit -m \"Initial commit\"")
    
    print("\n3. Create GitHub repository:")
    print("   - Go to https://github.com/new")
    print("   - Create repository")
    print("   - Follow push instructions")
    
    print("\n4. Upload models separately:")
    print("   - See MODEL_DOWNLOAD.md for options")
    print("   - Hugging Face Hub (recommended)")
    print("   - Google Drive")
    
    if large_files:
        print("\n⚠️ IMPORTANT: Large files detected!")
        print("   These need to be handled separately.")
        print("   See the file size check above.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()