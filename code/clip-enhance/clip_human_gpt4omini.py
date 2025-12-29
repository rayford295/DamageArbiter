import os, random, gc, traceback
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from scipy.stats import entropy

# 3) Set random seed & device
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4) Path and column configuration (modify if needed)
excel_path = "/content/drive/MyDrive/Manuscript_Ph.D./2nd_disasterCLIP (revise)/datatset/0310 dataset/0311_POST_cleaned.xlsx"
img_root   = "/content/drive/MyDrive/Manuscript_Ph.D./2nd_disasterCLIP (revise)/dataset/0310_dataset/0310_post_folder"

# Two types of descriptions: GPT vs Human
text_cols_for_comparison = ["summary_contrast", "txt_content"]  # GPT, Human
col2tag = {"summary_contrast": "GPT", "txt_content": "Human"}

# Label column
label_col           = "human_damage_perception"
processed_label_col = "label"
root_col            = "root"

# Class information
class_names = ["mild", "moderate", "severe"]
label_map   = {"mild": 0, "moderate": 1, "severe": 2}

# 5) Load & clean data
print("\n--- Loading and cleaning data ---")
if not os.path.exists(excel_path): raise FileNotFoundError(f"Data file not found: {excel_path}")
if not os.path.exists(img_root):   raise FileNotFoundError(f"Image root not found: {img_root}")

df = pd.read_excel(excel_path)
print(f"Raw shape: {df.shape}")

# Required column check
required_cols = text_cols_for_comparison + [label_col, root_col]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in data: {missing_cols}")

# Drop rows with missing text/directory
initial_len = len(df)
df = df.dropna(subset=text_cols_for_comparison + [root_col]).copy()
print(f"Dropped {initial_len - len(df)} rows due to NaNs in text/root.")

# Label cleaning
df[processed_label_col] = df[label_col].astype(str).str.lower().map(label_map)
initial_len = len(df)
df = df.dropna(subset=[processed_label_col]).copy()
df[processed_label_col] = df[processed_label_col].astype(int)
print(f"Dropped {initial_len - len(df)} rows due to unmappable labels.")
print("Label distribution:\n", df[processed_label_col].value_counts().sort_index())

# Pre-find the first image for each sample (improves robustness)
def find_first_image(folder):
    if not os.path.isdir(folder): return None
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not files: return None
    return os.path.join(folder, files[0])

df["first_image_path"] = df[root_col].astype(str).apply(lambda r: find_first_image(os.path.join(img_root, r)))
initial_len = len(df)
df = df.dropna(subset=["first_image_path"]).copy()
print(f"Dropped {initial_len - len(df)} rows without valid image file.")
print(f"Final usable samples: {len(df)}")

# 6) Dataset
class MultimodalDataset(Dataset):
    """
    - Use single text column for training/evaluation ('summary_contrast' or 'txt_content')
    - Pass a list of column names for CLIPScore calculation
    """
    def __init__(self, df, text_cols, processor, label_col=processed_label_col):
        self.df = df.reset_index(drop=True)
        self.text_cols = text_cols if isinstance(text_cols, (list, tuple)) else [text_cols]
        self.proc = processor
        self.label_col = label_col
        self.max_length = 77

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = row["first_image_path"]
        if not (isinstance(img_path, str) and os.path.isfile(img_path)):
            raise FileNotFoundError(f"Image not found for index {i}: {img_path}")

        img = Image.open(img_path).convert("RGB")
        image_inputs = self.proc(images=img, return_tensors="pt")
        item = {
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "path": img_path
        }
        for col in self.text_cols:
            text = str(row[col]) if pd.notna(row[col]) else ""
            text_inputs = self.proc(
                text=text, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            item[f"text_inputs_{col}"] = {
                "input_ids": text_inputs["input_ids"].squeeze(0),
                "attention_mask": text_inputs["attention_mask"].squeeze(0)
            }
        return item

# 7) Calculate CLIPScore (similarity between image and its own text)
def calculate_clip_scores(clip_model, processor, dataset, text_cols_to_score, batch_size=64, device="cpu"):
    print(f"\n--- Calculating CLIP Scores for: {text_cols_to_score} ---")
    clip_model.to(device).eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    all_scores = {col: [] for col in text_cols_to_score}

    with torch.no_grad():
        for batch in tqdm(dl, desc="CLIPScore"):
            pixel_values = batch["pixel_values"].to(device)
            img_feat = clip_model.get_image_features(pixel_values=pixel_values)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            for col_name in text_cols_to_score:
                tkey = f"text_inputs_{col_name}"
                input_ids = batch[tkey]["input_ids"].to(device)
                attn_mask = batch[tkey]["attention_mask"].to(device)
                txt_feat = clip_model.get_text_features(input_ids=input_ids, attention_mask=attn_mask)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                # Diagonal similarity with its own text
                sim = torch.diag(img_feat @ txt_feat.T)
                scores = 100 * sim
                all_scores[col_name].extend(scores.detach().cpu().tolist())

    avg_scores = {col: (np.mean(v) if len(v)>0 else 0.0) for col, v in all_scores.items()}
    print("CLIPScore Done:", {k: round(v,4) for k,v in avg_scores.items()})
    return avg_scores

# 8) CLIP Classification Head (ViT-B/16) -- Average of text and image features followed by a linear layer
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", num_classes=3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        feat_dim = self.clip.config.projection_dim  # Common dimension after image/text projection
        self.fc = nn.Linear(feat_dim, num_classes)
        print(f"Initialized CLIPClassifier base '{clip_model_name}', head={num_classes} classes.")

    def forward(self, pixel_values, input_ids, attention_mask=None):
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        txt_feat = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        fused = (img_feat + txt_feat) / 2.0
        logits = self.fc(fused)
        return logits

# 9) Training and Evaluation (Evaluation writes OOF & MIS)
def train_one_fold(train_ds, cfg, save_path):
    print(f"Training fold → save to: {save_path}")
    model = CLIPClassifier(clip_model_name=cfg["pretrained"], num_classes=cfg["num_classes"]).to(device)
    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    dl_tr = DataLoader(train_ds, batch_size=cfg["train_bsz"], shuffle=True,
                       pin_memory=True, num_workers=2)
    total_steps = max(1, len(dl_tr) * cfg["epochs"])
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, cfg["epochs"]+1):
        tr_loss, correct, total, steps = 0.0, 0, 0, 0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep}/{cfg['epochs']}")
        for batch in pbar:
            try:
                pixel_values = batch["pixel_values"].to(device)
                # Use the unique text key in this dataset
                txt_key = next(k for k in batch.keys() if k.startswith("text_inputs_"))
                input_ids = batch[txt_key]["input_ids"].to(device)
                attn_mask = batch[txt_key]["attention_mask"].to(device)
                labels    = batch["label"].to(device)

                opt.zero_grad()
                logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn_mask)
                loss   = crit(logits, labels)
                loss.backward()
                opt.step(); scheduler.step()

                tr_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                steps   += 1
                pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            except Exception as e:
                print(f"\n[TrainError] step failed: {e}")
                traceback.print_exc()
                break
        if steps>0:
            print(f"Epoch {ep}: loss={tr_loss/steps:.4f}  acc={correct/max(1,total):.4f}")
        else:
            print(f"Epoch {ep}: no steps processed.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved fold model to: {save_path}")
    return model

def evaluate(model, test_ds, cfg, save_dir, tag, fold_idx, class_names):
    print("Evaluating...")
    model.to(device).eval()
    dl_te = DataLoader(test_ds, batch_size=cfg["eval_bsz"], shuffle=False,
                       pin_memory=True, num_workers=1)
    all_preds, all_gts = [], []
    oof_rows = []

    with torch.no_grad():
        for batch in tqdm(dl_te, desc="Eval"):
            pixel_values = batch["pixel_values"].to(device)
            txt_key = next(k for k in batch.keys() if k.startswith("text_inputs_"))
            input_ids = batch[txt_key]["input_ids"].to(device)
            attn_mask = batch[txt_key]["attention_mask"].to(device)
            labels = batch["label"].numpy()
            paths  = batch["path"]

            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn_mask)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(1)

            all_preds.extend(preds.tolist())
            all_gts.extend(labels.tolist())

            for i in range(len(preds)):
                pvec = probs[i]
                ti, pi = int(labels[i]), int(preds[i])
                oof_rows.append({
                    "fold": fold_idx, "path": paths[i],
                    "true": class_names[ti], "pred": class_names[pi],
                    "probs_0": float(pvec[0]), "probs_1": float(pvec[1]), "probs_2": float(pvec[2]),
                    "p_true": float(pvec[ti]), "p_pred": float(pvec[pi]),
                    "confidence": float(pvec[pi] - pvec[ti]),
                    "entropy": float(entropy(pvec, base=2)),
                    "is_correct": int(ti == pi),
                })

    if len(all_gts)==0:
        metrics = {"Accuracy":0,"Precision":0,"Recall":0,"SW_F1":0,"MCC":0}
    else:
        metrics = {
            "Accuracy": accuracy_score(all_gts, all_preds),
            "Precision": precision_score(all_gts, all_preds, average="weighted", zero_division=0),
            "Recall": recall_score(all_gts, all_preds, average="weighted", zero_division=0),
            "SW_F1": f1_score(all_gts, all_preds, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(all_gts, all_preds),
        }

    # Write OOF & MIS
    os.makedirs(save_dir, exist_ok=True)
    oof_df = pd.DataFrame(oof_rows)
    oof_path = os.path.join(save_dir, f"oof_{tag.lower()}_fold{fold_idx}.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"Saved OOF → {oof_path}")

    mis = oof_df[oof_df["is_correct"]==0].copy()
    overconf_th, ambiguous_th = 0.40, 0.10
    mis["error_type"] = np.where(
        mis["confidence"]>=overconf_th, "overconfident",
        np.where(mis["confidence"]<=ambiguous_th, "ambiguous", "medium")
    )
    mis_path = os.path.join(save_dir, f"oof_{tag.lower()}_fold{fold_idx}_misclassified.csv")
    mis.to_csv(mis_path, index=False)
    print(f"Saved MIS  → {mis_path}")
    print("Misclassified type counts:\n", mis["error_type"].value_counts())
    return metrics, oof_path, mis_path

# 10) Configuration (Unified ViT-B/16)
cfg_clip_score = {
    "pretrained": "openai/clip-vit-base-patch16",
    "eval_bsz": 64,
}
cfg = {
    "pretrained": "openai/clip-vit-base-patch16",
    "epochs": 10,
    "train_bsz": 16,
    "eval_bsz": 32,
    "lr": 1e-5,
    "warmup_ratio": 0.1,
    "folds": 3,
    "seed": 42,
    "num_classes": 3,
    "weight_decay": 0.0
}

# 11) Calculate CLIPScore first (Optional)
print("\n--- CLIPScore (ViT-B/16) ---")
try:
    cs_processor = CLIPProcessor.from_pretrained(cfg_clip_score["pretrained"])
    cs_model     = CLIPModel.from_pretrained(cfg_clip_score["pretrained"])
    cs_dataset   = MultimodalDataset(df, text_cols_for_comparison, cs_processor, processed_label_col)
    clip_score_results = calculate_clip_scores(cs_model, cs_processor, cs_dataset,
                                               text_cols_to_score=text_cols_for_comparison,
                                               batch_size=cfg_clip_score["eval_bsz"], device=device)
except Exception as e:
    print("CLIPScore error:", e)
    clip_score_results = {}
finally:
    for var in ["cs_processor","cs_model","cs_dataset"]:
        if var in locals(): del globals()[var]
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# 12) Perform 3-fold fine-tuning using "Human/GPT descriptions" respectively, export weights and OOF
base_model_dir = "/content/saved_classifier_models"        # Weight output
base_oof_dir   = "/content/oof_clip_classifier"            # OOF/MIS output
os.makedirs(base_model_dir, exist_ok=True)
os.makedirs(base_oof_dir, exist_ok=True)

results = {}  # Record per-fold and avg metrics for each text type
ft_processor = CLIPProcessor.from_pretrained(cfg["pretrained"])

for col in text_cols_for_comparison:
    tag = col2tag.get(col, col)  # "GPT"/"Human"
    print(f"\n{'='*18} FINE-TUNING CV for [{tag}] ({col}) {'='*18}")

    skf = StratifiedKFold(n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"])
    fold_metrics = []
    model_dir = os.path.join(base_model_dir, tag)
    oof_dir   = os.path.join(base_oof_dir, tag)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(oof_dir, exist_ok=True)
    print(f"Models → {model_dir}")
    print(f"OOFs   → {oof_dir}")

    for k, (tr_idx, te_idx) in enumerate(skf.split(df, df[processed_label_col]), start=1):
        print(f"\n--- Fold {k}/{cfg['folds']} ---")
        tr_df, te_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        print(f"Train={len(tr_df)}  Test={len(te_df)}")
        print("Train dist:\n", tr_df[processed_label_col].value_counts().sort_index())
        print("Test  dist:\n", te_df[processed_label_col].value_counts().sort_index())

        ds_tr = MultimodalDataset(tr_df, col, ft_processor, processed_label_col)
        ds_te = MultimodalDataset(te_df, col, ft_processor, processed_label_col)

        model_path = os.path.join(model_dir, f"clip_classifier_{tag.lower()}_fold{k}.pt")
        model = train_one_fold(ds_tr, cfg, model_path)

        metrics, oof_path, mis_path = evaluate(model, ds_te, cfg, oof_dir, tag, k, class_names)
        print(f"Fold {k} Metrics ({tag}): {metrics}")
        fold_metrics.append(metrics)

        # Release GPU memory
        del model, ds_tr, ds_te
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Aggregate averages
    if fold_metrics:
        avg = {m: float(np.mean([x[m] for x in fold_metrics])) for m in fold_metrics[0].keys()}
    else:
        avg = {}
    results[tag] = {"per_fold": fold_metrics, "avg": avg}

# 13) Print final results and output locations
print("\n" + "="*52)
print("                 EXECUTION COMPLETE")
print("="*52)

print("\n--- CLIPScore (image ↔ text) ---")
if clip_score_results:
    for col, v in clip_score_results.items():
        print(f"  {col2tag.get(col, col)}: {v:.4f}")
else:
    print("  (Skipped or failed)")

print("\n--- Fine-tuning Results ---")
for tag, res in results.items():
    print(f"\n[{tag}]")
    if res["per_fold"]:
        for i, m in enumerate(res["per_fold"], start=1):
            print(f"  Fold {i}: " + ", ".join([f"{k}={m[k]:.4f}" for k in m]))
        if res["avg"]:
            print("  Avg : " + ", ".join([f"{k}={res['avg'][k]:.4f}" for k in res["avg"]]))
    else:
        print("  (No metrics)")

print(f"\nModel weights saved under: {base_model_dir}")
print(f"OOF & misclassified CSVs under: {base_oof_dir}")
