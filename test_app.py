"""
Taglish Gaslighting Detection - Interactive Testing App V3
===========================================================

A Gradio web interface to test your trained models on custom text inputs.
Uses sequential classification: Binary detection → Tactic identification

Features:
- Single text analysis with sequential pipeline
- Batch processing for CSV files
- Model performance comparison
- Comprehensive visualizations

Usage:
    python test_app.py

Then open the URL shown in your browser (usually http://127.0.0.1:7860)
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """App configuration"""
    
    # Model paths - UPDATE TO V3
    BASE_DIR = "D:/Thesis/Taglish_Gaslighting_V3"
    MODEL_DIR = os.path.join(BASE_DIR, "model_outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
    
    # Available models (3 models × 2 classifiers each)
    MODELS = {
        "roberta-tagalog": {
            "display": "RoBERTa-Tagalog",
            "binary_path": os.path.join(MODEL_DIR, "roberta-tagalog_binary"),
            "tactic_path": os.path.join(MODEL_DIR, "roberta-tagalog_tactic"),
            "performance": {
                "binary_f1_id": 0.9291,
                "binary_f1_ood": 0.8917,
                "binary_roc_auc": 0.9802,
                "tactic_f1_id": 0.9085,
                "tactic_f1_ood": 0.5872,
                "binary_delta_f1": 0.0374,
                "tactic_delta_f1": 0.3214
            }
        },
        "mbert": {
            "display": "mBERT",
            "binary_path": os.path.join(MODEL_DIR, "mbert_binary"),
            "tactic_path": os.path.join(MODEL_DIR, "mbert_tactic"),
            "performance": {
                "binary_f1_id": 0.9053,
                "binary_f1_ood": 0.9107,
                "binary_roc_auc": 0.9741,
                "tactic_f1_id": 0.8283,
                "tactic_f1_ood": 0.5007,
                "binary_delta_f1": -0.0054,
                "tactic_delta_f1": 0.3276
            }
        },
        "xlm-roberta": {
            "display": "XLM-RoBERTa",
            "binary_path": os.path.join(MODEL_DIR, "xlm-roberta_binary"),
            "tactic_path": os.path.join(MODEL_DIR, "xlm-roberta_tactic"),
            "performance": {
                "binary_f1_id": 0.8897,
                "binary_f1_ood": 0.8697,
                "binary_roc_auc": 0.9518,
                "tactic_f1_id": 0.7582,
                "tactic_f1_ood": 0.4761,
                "binary_delta_f1": 0.0200,
                "tactic_delta_f1": 0.2821
            }
        }
    }
    
    # Label mappings
    BINARY_LABELS = {
        0: "Non-Gaslighting",
        1: "Gaslighting"
    }
    
    TACTIC_LABELS = {
        0: "Distortion & Denial",
        1: "Trivialization & Minimization",
        2: "Coercion & Intimidation",
        3: "Knowledge Invalidation"
    }
    
    TACTIC_DESCRIPTIONS = {
        0: "**Distortion & Denial**: Denying or distorting facts, spreading misinformation, or rewriting history.",
        1: "**Trivialization & Minimization**: Minimizing concerns, dismissing feelings, or making issues seem overdramatic.",
        2: "**Coercion & Intimidation**: Using threats, intimidation, or coercion to control or silence someone.",
        3: "**Knowledge Invalidation**: Attacking credibility, intelligence, or knowledge to undermine someone."
    }
    
    TACTIC_EMOJIS = {
        0: "🔴",
        1: "🟡",
        2: "🟠",
        3: "🔵"
    }
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# MODEL LOADER
# ==========================================

class ModelCache:
    """Cache loaded models to avoid reloading"""
    
    def __init__(self):
        self.cache = {}
    
    def load_model_pair(self, model_key):
        """Load both binary and tactic models for a given model key"""
        
        if model_key in self.cache:
            return self.cache[model_key]
        
        model_info = Config.MODELS[model_key]
        
        # Check if models exist
        if not os.path.exists(model_info["binary_path"]):
            raise FileNotFoundError(
                f"Binary model not found at {model_info['binary_path']}\n"
                f"Please ensure you've trained the models first."
            )
        
        if not os.path.exists(model_info["tactic_path"]):
            raise FileNotFoundError(
                f"Tactic model not found at {model_info['tactic_path']}\n"
                f"Please ensure you've trained the models first."
            )
        
        print(f"Loading {model_info['display']} models...")
        
        # Load binary classifier
        binary_tokenizer = AutoTokenizer.from_pretrained(model_info["binary_path"])
        binary_model = AutoModelForSequenceClassification.from_pretrained(model_info["binary_path"])
        binary_model.to(Config.DEVICE)
        binary_model.eval()
        
        # Load tactic classifier
        tactic_tokenizer = AutoTokenizer.from_pretrained(model_info["tactic_path"])
        tactic_model = AutoModelForSequenceClassification.from_pretrained(model_info["tactic_path"])
        tactic_model.to(Config.DEVICE)
        tactic_model.eval()
        
        # Cache
        self.cache[model_key] = {
            "binary": {
                "tokenizer": binary_tokenizer,
                "model": binary_model
            },
            "tactic": {
                "tokenizer": tactic_tokenizer,
                "model": tactic_model
            },
            "info": model_info
        }
        
        print(f"✓ {model_info['display']} loaded successfully")
        
        return self.cache[model_key]

# Initialize cache
model_cache = ModelCache()

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def predict_sequential(text, model_key):
    """
    Sequential prediction: Binary → Tactic (if gaslighting)
    
    Args:
        text: Input text to analyze
        model_key: Key of model to use
    
    Returns:
        Formatted prediction results
    """
    
    if not text or not text.strip():
        return "⚠️ Please enter some text to analyze.", None, None
    
    try:
        # Load models
        models = model_cache.load_model_pair(model_key)
        binary_tokenizer = models["binary"]["tokenizer"]
        binary_model = models["binary"]["model"]
        tactic_tokenizer = models["tactic"]["tokenizer"]
        tactic_model = models["tactic"]["model"]
        model_info = models["info"]
        
        # ===== STEP 1: Binary Classification =====
        binary_inputs = binary_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        binary_inputs = {k: v.to(Config.DEVICE) for k, v in binary_inputs.items()}
        
        with torch.no_grad():
            binary_outputs = binary_model(**binary_inputs)
            binary_logits = binary_outputs.logits
            binary_probs = torch.softmax(binary_logits, dim=-1)
            binary_pred = torch.argmax(binary_probs, dim=-1).item()
            binary_confidence = binary_probs[0][binary_pred].item()
        
        # Binary result
        is_gaslighting = (binary_pred == 1)
        binary_label = Config.BINARY_LABELS[binary_pred]
        
        # Create binary probability dict
        binary_prob_dict = {
            "Non-Gaslighting": float(binary_probs[0][0]),
            "Gaslighting": float(binary_probs[0][1])
        }
        
        # ===== STEP 2: Tactic Classification (if gaslighting) =====
        tactic_result = None
        tactic_prob_dict = None
        
        if is_gaslighting:
            tactic_inputs = tactic_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            tactic_inputs = {k: v.to(Config.DEVICE) for k, v in tactic_inputs.items()}
            
            with torch.no_grad():
                tactic_outputs = tactic_model(**tactic_inputs)
                tactic_logits = tactic_outputs.logits
                tactic_probs = torch.softmax(tactic_logits, dim=-1)
                tactic_pred = torch.argmax(tactic_probs, dim=-1).item()
                tactic_confidence = tactic_probs[0][tactic_pred].item()
            
            tactic_label = Config.TACTIC_LABELS[tactic_pred]
            tactic_emoji = Config.TACTIC_EMOJIS[tactic_pred]
            tactic_description = Config.TACTIC_DESCRIPTIONS[tactic_pred]
            
            # Create tactic probability dict
            tactic_prob_dict = {
                Config.TACTIC_LABELS[i]: float(tactic_probs[0][i]) 
                for i in range(4)
            }
            
            # Format tactic result
            tactic_result = f"""
### {tactic_emoji} Gaslighting Tactic: {tactic_label}

**Confidence:** {tactic_confidence:.1%}

{tactic_description}
"""
        
        # ===== Format Final Output =====
        
        if is_gaslighting:
            status_icon = "⚠️"
            status_color = "red"
        else:
            status_icon = "✅"
            status_color = "green"
        
        result = f"""
# {status_icon} Detection Result

## Binary Classification
**Prediction:** {binary_label}  
**Confidence:** {binary_confidence:.1%}

---

{tactic_result if tactic_result else "**No tactic classification needed** - Text is classified as non-gaslighting."}

---

## Model Performance ({model_info['display']})

**Binary Classification:**
- In-Domain F1: {model_info['performance']['binary_f1_id']:.3f}
- Out-of-Domain F1: {model_info['performance']['binary_f1_ood']:.3f}
- ROC-AUC: {model_info['performance']['binary_roc_auc']:.3f}
- ΔF1: {model_info['performance']['binary_delta_f1']:.3f}

**Tactic Classification:**
- In-Domain Macro-F1: {model_info['performance']['tactic_f1_id']:.3f}
- Out-of-Domain Macro-F1: {model_info['performance']['tactic_f1_ood']:.3f}
- ΔF1: {model_info['performance']['tactic_delta_f1']:.3f}
"""
        
        return result, binary_prob_dict, tactic_prob_dict
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None

def batch_predict(file, model_key):
    """
    Make sequential predictions on a CSV file
    
    Args:
        file: Uploaded CSV file with 'sentence' column
        model_key: Key of model to use
    
    Returns:
        DataFrame with predictions
    """
    
    try:
        # Read CSV
        df = pd.read_csv(file.name)
        
        if 'sentence' not in df.columns:
            return pd.DataFrame({"Error": ["CSV must have a 'sentence' column"]})
        
        # Load models
        models = model_cache.load_model_pair(model_key)
        binary_tokenizer = models["binary"]["tokenizer"]
        binary_model = models["binary"]["model"]
        tactic_tokenizer = models["tactic"]["tokenizer"]
        tactic_model = models["tactic"]["model"]
        
        binary_predictions = []
        binary_confidences = []
        tactic_predictions = []
        tactic_confidences = []
        
        # Process each sentence
        for text in df['sentence']:
            text = str(text)
            
            # Binary classification
            binary_inputs = binary_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            binary_inputs = {k: v.to(Config.DEVICE) for k, v in binary_inputs.items()}
            
            with torch.no_grad():
                binary_outputs = binary_model(**binary_inputs)
                binary_logits = binary_outputs.logits
                binary_probs = torch.softmax(binary_logits, dim=-1)
                binary_pred = torch.argmax(binary_probs, dim=-1).item()
                binary_conf = binary_probs[0][binary_pred].item()
            
            binary_label = Config.BINARY_LABELS[binary_pred]
            binary_predictions.append(binary_label)
            binary_confidences.append(f"{binary_conf:.1%}")
            
            # Tactic classification (if gaslighting)
            if binary_pred == 1:
                tactic_inputs = tactic_tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                tactic_inputs = {k: v.to(Config.DEVICE) for k, v in tactic_inputs.items()}
                
                with torch.no_grad():
                    tactic_outputs = tactic_model(**tactic_inputs)
                    tactic_logits = tactic_outputs.logits
                    tactic_probs = torch.softmax(tactic_logits, dim=-1)
                    tactic_pred = torch.argmax(tactic_probs, dim=-1).item()
                    tactic_conf = tactic_probs[0][tactic_pred].item()
                
                tactic_label = Config.TACTIC_LABELS[tactic_pred]
                tactic_predictions.append(tactic_label)
                tactic_confidences.append(f"{tactic_conf:.1%}")
            else:
                tactic_predictions.append("N/A")
                tactic_confidences.append("N/A")
        
        # Add predictions to dataframe
        df['binary_prediction'] = binary_predictions
        df['binary_confidence'] = binary_confidences
        df['tactic_prediction'] = tactic_predictions
        df['tactic_confidence'] = tactic_confidences
        
        return df
        
    except Exception as e:
        import traceback
        return pd.DataFrame({"Error": [str(e)], "Traceback": [traceback.format_exc()]})

# ==========================================
# EXAMPLE TEXTS
# ==========================================

EXAMPLES = {
    "Political - Distortion": "Walang nangyaring martial law abuses. Propaganda lang yan ng mga dilawan at western media.",
    "Political - Trivialization": "OA ka naman, konting problema lang pinapalaki mo. Move on ka na.",
    "Political - Coercion": "Kung ipost mo yan, lagot ka. Ipapahiya kita at ikalat ko lahat ng info mo.",
    "Political - Knowledge Inv.": "Bobo ka ba? Wala kang alam sa politics. Di mo maintindihan ang tunay na nangyayari.",
    "Cultural - Family": "Walang utang na loob ka. Lahat ginawa namin para sayo tapos ganyan ka lang?",
    "Cultural - Gender": "Babae ka, dapat sumunod ka lang. Yan ang kultura natin, huwag kang maarte.",
    "Revisionist - Martial Law": "Golden age ng Pilipinas yan. Walang namatay, gawa-gawa lang ng media.",
    "Non-Gaslighting - Critique": "I respectfully disagree with this policy. Here are my concerns based on data.",
    "Non-Gaslighting - Support": "Thank you for sharing this. I appreciate your perspective on this issue."
}

# ==========================================
# GRADIO INTERFACE
# ==========================================

def create_interface():
    """Create Gradio web interface"""
    
    # Custom CSS
    css = """
    /* TARGETS THE RESULT TEXT */
    .prediction-output {
        font-size: 24px !important;       /* Increased from 16px */
        line-height: 1.8 !important;      /* Better spacing between lines */
        padding: 40px !important;         /* HUGE internal margins */
        margin-top: 20px !important;      /* Space above the box */
        border: 1px solid #e0e0e0;        /* Optional: adds a border */
        border-radius: 10px;              /* Optional: rounds the corners */
        background-color: #f9f9f9;        /* Optional: light background */
    }
    .prediction-output {
        font-size: 16px;
        line-height: 1.8;
    }
    .gradio-container {
        max-width: 1400px;
        margin: auto;
    }
    .model-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(title="Taglish Gaslighting Detector V3", theme=gr.themes.Soft()) as app:
        
        # Apply custom CSS
        gr.HTML(f"<style>{css}</style>")
        
        gr.Markdown("""
        # 🔍 Taglish Gaslighting Detection System V3
        
        **Sequential Classification Pipeline: Binary Detection → Tactic Identification**
        
        This system uses a two-stage approach:
        1. **Binary Classifier** determines if text contains gaslighting
        2. **Tactic Classifier** identifies the specific manipulation tactic (if gaslighting detected)
        
        All models trained on Philippine political discourse with zero-shot evaluation on cultural and revisionist domains.
        """)
        
        with gr.Tabs():
            
            # ===== SINGLE TEXT PREDICTION =====
            with gr.Tab("🔍 Single Text Analysis"):
                
                gr.Markdown("### Analyze individual text samples with sequential classification")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="Type or paste Tagalog/Taglish text here...\n\nExample: 'OA ka naman, konting problema lang yan.'",
                            lines=6
                        )
                        
                        model_dropdown = gr.Dropdown(
                            choices=[
                                ("RoBERTa-Tagalog (Best In-Domain)", "roberta-tagalog"),
                                ("mBERT (Best OOD Generalization)", "mbert"),
                                ("XLM-RoBERTa (Strong Multilingual)", "xlm-roberta")
                            ],
                            value="roberta-tagalog",
                            label="Select Model",
                            info="Each model has both binary and tactic classifiers"
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                        
                        gr.Markdown("---")
                        
                        # Model info
                        gr.Markdown("""
                        **Model Selection Guide:**
                        - **RoBERTa-Tagalog**: Best for Tagalog-heavy political text
                        - **mBERT**: Best cross-domain generalization
                        - **XLM-RoBERTa**: Strong on code-switched text
                        """)
                    
                    with gr.Column(scale=2):
                        prediction_output = gr.Markdown(
                            label="Analysis Results",
                            elem_classes=["prediction-output"]
                        )
                        
                        with gr.Row():
                            binary_plot = gr.Label(
                                label="Binary Classification Probabilities",
                                num_top_classes=2
                            )
                            
                            tactic_plot = gr.Label(
                                label="Tactic Classification Probabilities (if gaslighting)",
                                num_top_classes=4
                            )
                
                # Examples
                gr.Markdown("### 📝 Example Texts (Click to try)")
                
                with gr.Row():
                    for name in list(EXAMPLES.keys())[:3]:
                        gr.Button(name, size="sm").click(
                            fn=lambda n=name: EXAMPLES[n],
                            inputs=None,
                            outputs=text_input
                        )
                
                with gr.Row():
                    for name in list(EXAMPLES.keys())[3:6]:
                        gr.Button(name, size="sm").click(
                            fn=lambda n=name: EXAMPLES[n],
                            inputs=None,
                            outputs=text_input
                        )
                
                with gr.Row():
                    for name in list(EXAMPLES.keys())[6:]:
                        gr.Button(name, size="sm").click(
                            fn=lambda n=name: EXAMPLES[n],
                            inputs=None,
                            outputs=text_input
                        )
                
                # Wire up prediction
                analyze_btn.click(
                    fn=predict_sequential,
                    inputs=[text_input, model_dropdown],
                    outputs=[prediction_output, binary_plot, tactic_plot]
                )
            
            # ===== BATCH PREDICTION =====
            with gr.Tab("📊 Batch Processing"):
                
                gr.Markdown("""
                ### Process multiple texts from CSV
                
                Upload a CSV file with a `sentence` column. The system will:
                1. Run binary classification on all samples
                2. Run tactic classification on samples identified as gaslighting
                3. Export results with both predictions
                
                **Output columns:**
                - `binary_prediction`: Gaslighting or Non-Gaslighting
                - `binary_confidence`: Confidence score
                - `tactic_prediction`: Specific tactic (or N/A if non-gaslighting)
                - `tactic_confidence`: Confidence score
                """)
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"]
                        )
                        
                        batch_model = gr.Dropdown(
                            choices=[
                                ("RoBERTa-Tagalog", "roberta-tagalog"),
                                ("mBERT", "mbert"),
                                ("XLM-RoBERTa", "xlm-roberta")
                            ],
                            value="roberta-tagalog",
                            label="Select Model"
                        )
                        
                        process_btn = gr.Button("📊 Process Batch", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **CSV Format:**
                        ```
                        sentence
                        "Your first text here"
                        "Your second text here"
                        ```
                        """)
                    
                    with gr.Column():
                        batch_output = gr.Dataframe(
                            label="Results Preview",
                            wrap=True,
                            interactive=False
                        )
                        
                        download_btn = gr.File(label="Download Complete Results")
                
                # Wire up batch processing
                def process_and_save(file, model):
                    if file is None:
                        return pd.DataFrame({"Error": ["Please upload a file"]}), None
                    
                    results_df = batch_predict(file, model)
                    
                    # Save to temp file
                    output_path = "batch_predictions.csv"
                    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    return results_df, output_path
                
                process_btn.click(
                    fn=process_and_save,
                    inputs=[file_input, batch_model],
                    outputs=[batch_output, download_btn]
                )
            
            # ===== MODEL PERFORMANCE =====
            with gr.Tab("📈 Model Performance"):
                
                gr.Markdown("## 📊 Comprehensive Evaluation Results")
                
                # Binary Classification Table
                gr.Markdown("### Binary Classification Performance")
                
                binary_data = []
                for key, info in Config.MODELS.items():
                    perf = info['performance']
                    binary_data.append({
                        "Model": info['display'],
                        "In-Domain F1": f"{perf['binary_f1_id']:.4f}",
                        "OOD F1": f"{perf['binary_f1_ood']:.4f}",
                        "ROC-AUC": f"{perf['binary_roc_auc']:.4f}",
                        "ΔF1": f"{perf['binary_delta_f1']:.4f}",
                        "Generalization": "✅ Robust" if abs(perf['binary_delta_f1']) < 0.10 else "⚠️ Moderate"
                    })
                
                gr.Dataframe(
                    pd.DataFrame(binary_data),
                    wrap=True
                )
                
                gr.Markdown("""
                **Key Findings - Binary Classification:**
                - All models achieve >88% F1 score on in-domain test
                - mBERT shows best cross-domain generalization (negative ΔF1 = better OOD than in-domain!)
                - RoBERTa-Tagalog achieves highest in-domain performance (92.9% F1)
                - All models show robust generalization (ΔF1 < 0.10)
                """)
                
                # Tactic Classification Table
                gr.Markdown("### Tactic Classification Performance (4-Class)")
                
                tactic_data = []
                for key, info in Config.MODELS.items():
                    perf = info['performance']
                    tactic_data.append({
                        "Model": info['display'],
                        "In-Domain Macro-F1": f"{perf['tactic_f1_id']:.4f}",
                        "OOD Macro-F1": f"{perf['tactic_f1_ood']:.4f}",
                        "ΔF1": f"{perf['tactic_delta_f1']:.4f}",
                        "Generalization": "❌ Significant" if perf['tactic_delta_f1'] >= 0.15 else "⚠️ Moderate"
                    })
                
                gr.Dataframe(
                    pd.DataFrame(tactic_data),
                    wrap=True
                )
                
                gr.Markdown("""
                **Key Findings - Tactic Classification:**
                - Tactic classification is significantly more challenging than binary
                - All models show substantial performance drop on OOD data (ΔF1 ≥ 0.28)
                - RoBERTa-Tagalog maintains best performance (90.9% in-domain, 58.7% OOD)
                - Domain adaptation techniques recommended for improved cross-domain tactic detection
                
                **ΔF1 Interpretation:**
                - ΔF1 < 0.10: ✅ Robust generalization
                - ΔF1 0.10-0.15: ⚠️ Moderate domain dependence  
                - ΔF1 ≥ 0.15: ❌ Significant overfitting
                """)
                
                gr.Markdown("---")
                
                # Model Comparison
                gr.Markdown("### 🔬 Detailed Model Comparison")
                
                for key, info in Config.MODELS.items():
                    with gr.Accordion(f"{info['display']} - Detailed Performance", open=False):
                        perf = info['performance']
                        
                        gr.Markdown(f"""
                        **{info['display']}**
                        
                        **Binary Classification:**
                        - In-Domain F1: {perf['binary_f1_id']:.4f}
                        - Out-of-Domain F1: {perf['binary_f1_ood']:.4f}
                        - ROC-AUC: {perf['binary_roc_auc']:.4f}
                        - Domain Shift (ΔF1): {perf['binary_delta_f1']:.4f}
                        
                        **Tactic Classification:**
                        - In-Domain Macro-F1: {perf['tactic_f1_id']:.4f}
                        - Out-of-Domain Macro-F1: {perf['tactic_f1_ood']:.4f}
                        - Domain Shift (ΔF1): {perf['tactic_delta_f1']:.4f}
                        
                        **Strengths:**
                        {_get_model_strengths(key)}
                        
                        **Limitations:**
                        {_get_model_limitations(key)}
                        """)
            
            # ===== GASLIGHTING TACTICS GUIDE =====
            with gr.Tab("📖 Tactics Guide"):
                
                gr.Markdown("""
                ## Understanding Gaslighting Tactics
                
                This system identifies 4 main types of gaslighting tactics in Taglish discourse:
                """)
                
                with gr.Accordion("🔴 Distortion & Denial", open=True):
                    gr.Markdown("""
                    **Definition:** Denying or distorting facts, spreading misinformation, or rewriting history.
                    
                    **Examples:**
                    - "Walang nangyaring martial law abuses. Propaganda lang yan."
                    - "Fake news yan, gawa-gawa lang ng media."
                    - "Golden age ng Pilipinas yan, wala talagang namatay."
                    
                    **Key Indicators:**
                    - Denial of documented facts
                    - Rewriting historical events
                    - Dismissing evidence as "propaganda" or "fake news"
                    """)
                
                with gr.Accordion("🟡 Trivialization & Minimization", open=False):
                    gr.Markdown("""
                    **Definition:** Minimizing someone's concerns, dismissing their feelings, or making issues seem overdramatic.
                    
                    **Examples:**
                    - "OA ka naman, konting problema lang yan."
                    - "Malaki ka na, dapat matured ka na. Wag kang balat sibuyas."
                    - "Drama mo naman, move on ka na."
                    
                    **Key Indicators:**
                    - Dismissing valid concerns as "overreacting"
                    - Making someone feel their feelings are invalid
                    - Telling someone to "move on" from serious issues
                    """)
                
                with gr.Accordion("🟠 Coercion & Intimidation", open=False):
                    gr.Markdown("""
                    **Definition:** Using threats, intimidation, or coercion to control or silence someone.
                    
                    **Examples:**
                    - "Kung ipost mo yan, lagot ka. Ipapahiya kita."
                    - "Manahimik ka na lang kung ayaw mong may mangyari sayo."
                    - "Alam mo ba kung sino kaaway mo? Wag kang magsalita."
                    
                    **Key Indicators:**
                    - Direct or implied threats
                    - Intimidation to silence dissent
                    - Using fear to control behavior
                    """)
                
                with gr.Accordion("🔵 Knowledge Invalidation", open=False):
                    gr.Markdown("""
                    **Definition:** Attacking someone's credibility, intelligence, or knowledge to undermine them.
                    
                    **Examples:**
                    - "Bobo ka ba? Wala kang alam sa politics."
                    - "Di mo naman maintindihan yan, walang pinag-aralan."
                    - "Ano ba alam mo? Wala kang karapatang magsalita."
                    
                    **Key Indicators:**
                    - Attacking intelligence or education
                    - Questioning someone's right to have an opinion
                    - Undermining credibility instead of addressing arguments
                    """)
                
                gr.Markdown("""
                ---
                
                ### 🎯 How to Use This Knowledge
                
                1. **Self-Protection:** Recognize when these tactics are being used against you
                2. **Content Moderation:** Identify manipulative discourse in online spaces
                3. **Research:** Analyze patterns of manipulation in political discourse
                4. **Education:** Teach others to recognize gaslighting behavior
                
                ### ⚠️ Important Notes
                
                - Context matters: Not all criticism is gaslighting
                - Cultural nuances: Some phrases may be normal discourse in certain contexts
                - Intent vs. Impact: Focus on the effect of the language, not just intent
                - Seek human judgment: This tool aids analysis but shouldn't replace human understanding
                """)
            
            # ===== ABOUT =====
            with gr.Tab("ℹ️ About"):
                
                gr.Markdown("""
                ## About This System
                
                ### 🎓 Research Context
                
                This gaslighting detection system was developed as part of a thesis on detecting manipulation 
                in Taglish political discourse. The system uses a **sequential classification pipeline**:
                
                1. **Binary Classifier**: Determines if text contains gaslighting (2 classes)
                2. **Tactic Classifier**: Identifies specific manipulation tactic (4 classes)
                
                ### 🤖 Models
                
                **Three transformer-based models, each with binary + tactic classifiers:**
                
                1. **RoBERTa-Tagalog** (`jcblaise/roberta-tagalog-base`)
                   - Monolingual Filipino model
                   - Best in-domain performance
                   - Optimized for Tagalog-heavy text
                
                2. **mBERT** (`bert-base-multilingual-uncased`)
                   - Multilingual baseline (100+ languages)
                   - Best cross-domain generalization
                   - Balanced performance across domains
                
                3. **XLM-RoBERTa** (`xlm-roberta-base`)
                   - Strong multilingual model
                   - Excellent for code-switched text
                   - Good OOD generalization
                
                ### 📊 Training Data
                
                **In-Domain (Training & Testing):**
                - Source: Philippine political discourse (Reddit)
                - Domain: Political gaslighting
                - Language: Taglish (Tagalog-English code-switching)
                
                **Out-of-Domain (Zero-Shot Testing):**
                - Cultural gaslighting (family, gender, tradition)
                - Historical revisionism (martial law, propaganda)
                - No training data from these domains
                
                ### 📈 Evaluation Metrics
                
                **Primary Metrics:**
                - Precision, Recall, F1-Score (per class)
                - Macro-F1 (equal weight to all classes)
                - ROC-AUC (binary classification)
                
                **Domain Generalization:**
                - ΔF1 = F1_in-domain - F1_out-of-domain
                - Measures performance degradation across domains
                
                ### ⚙️ Technical Details
                
                - **Framework**: PyTorch + Hugging Face Transformers
                - **Max Sequence Length**: 128 tokens
                - **Training**: 5 epochs with early stopping
                - **Hardware**: GPU-accelerated (CUDA)
                - **Optimization**: AdamW with warmup
                
                ### 🎯 Use Cases
                
                ✅ **Appropriate Uses:**
                - Content moderation assistance
                - Social media monitoring
                - Research on online manipulation
                - Educational tool for identifying gaslighting
                - Supporting human moderators
                
                ❌ **Inappropriate Uses:**
                - Sole basis for content removal decisions
                - Automated censorship without review
                - Personal conflict resolution
                - Legal proceedings without expert review
                
                ### ⚠️ Limitations
                
                - Trained primarily on political discourse
                - Performance degrades on unseen domains
                - Context and intent not fully captured
                - Cultural nuances may be missed
                - Not a substitute for human judgment
                - May have biases from training data
                
                ### 📚 Citation
                
                If you use this system in your research, please cite:
                
                ```
                [Your Name] ([Year]). Taglish Gaslighting Detection Using Transformer Models.
                [Your Institution]. [Thesis/Publication Details]
                ```
                
                ### 📧 Contact
                
                For questions, feedback, or collaboration opportunities:
                - Email: [Your Email]
                - Institution: [Your Institution]
                - GitHub: [Your GitHub]
                
                ---
                
                **Version**: 3.0  
                **Last Updated**: February 2026  
                **License**: [Your License]
                """)
        
        gr.Markdown("""
        ---
        
        **Disclaimer**: This is a research prototype. Results should be interpreted with care and in context. 
        Always exercise human judgment and consider cultural nuances when analyzing discourse.
        """)
    
    return app

def _get_model_strengths(model_key):
    """Get model-specific strengths"""
    strengths = {
        "roberta-tagalog": """
        - Highest in-domain performance on both tasks
        - Best tactic classification (90.9% Macro-F1)
        - Optimized for Filipino language processing
        - Strong on political discourse
        """,
        "mbert": """
        - Best OOD generalization on binary task
        - Balanced performance across domains
        - Handles multilingual text well
        - Robust to language variations
        """,
        "xlm-roberta": """
        - Strong on code-switched text
        - Good OOD performance
        - Handles diverse linguistic patterns
        - Moderate domain dependence
        """
    }
    return strengths.get(model_key, "")

def _get_model_limitations(model_key):
    """Get model-specific limitations"""
    limitations = {
        "roberta-tagalog": """
        - Largest OOD performance drop on tactic task
        - May struggle with pure English text
        - More domain-specific (less generalizable)
        """,
        "mbert": """
        - Lower in-domain performance than RoBERTa
        - Significant tactic classification OOD drop
        - May miss Tagalog-specific nuances
        """,
        "xlm-roberta": """
        - Lowest in-domain performance on both tasks
        - Largest tactic classification challenges
        - May require more training data
        """
    }
    return limitations.get(model_key, "")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("TAGLISH GASLIGHTING DETECTION - INTERACTIVE APP V3")
    print("="*70)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Model Directory: {Config.MODEL_DIR}")
    
    # Check if models exist
    print("\nChecking for trained models...")
    models_found = 0
    for key, info in Config.MODELS.items():
        binary_exists = os.path.exists(info["binary_path"])
        tactic_exists = os.path.exists(info["tactic_path"])
        
        if binary_exists and tactic_exists:
            print(f"  ✓ {info['display']} (Binary + Tactic)")
            models_found += 1
        else:
            print(f"  ✗ {info['display']}")
            if not binary_exists:
                print(f"      Missing: {info['binary_path']}")
            if not tactic_exists:
                print(f"      Missing: {info['tactic_path']}")
    
    if models_found == 0:
        print("\n❌ No models found! Please train models first using:")
        print("   python taglish_gaslighting_training_final.py")
        exit(1)
    
    print(f"\n✓ Found {models_found}/{len(Config.MODELS)} model pairs")
    print("\nStarting Gradio app...")
    print("="*70 + "\n")
    
    # Create and launch app
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )