# %% JUPYTER ALL-IN-ONE — Gemini API Image Processing
import os
import sys
import csv
import json
import traceback
import re
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types

# --- 1. Configuration Paths (Please ensure path exists) ---
# Your image root directory
BASE_DIR = r"C:\Users\yyang295\Desktop\0310_post_folder"   
FOLDER_TO_LABEL = {"folder_0": "mild", "folder_1": "moderate", "folder_2": "severe"}
CSV_NAME = "post_image_llm_labels.csv"
ERR_LOG  = os.path.join(BASE_DIR, "post_image_llm_errors.txt")

# --- 2. Configure API Key (Aggressive cleaning added) ---
# The Key that passed the test earlier
RAW_KEY = "..."

# Aggressive cleaning: Remove all spaces, tabs, newlines to prevent errors
CLEAN_KEY = re.sub(r'\s+', '', RAW_KEY).strip()

# Initialize client
client = genai.Client(api_key=CLEAN_KEY)

# Model name (The version that passed the test earlier)
MODEL_NAME = "gemini-2.0-flash-exp"
TEMPERATURE = 0.1

# --- 3. Configure Prompts ---
HEADLINE_MAX_WORDS = 12
SHORT_CAPTION_MAX_WORDS = 20
LONG_CAPTION_MIN_WORDS = 35
LONG_CAPTION_MAX_WORDS = 50

SYSTEM_INSTRUCTION = (
    "You are a careful disaster-damage annotator. "
    "Only describe evidence actually visible in the image. Avoid hallucinations or speculation."
)

def build_prompt() -> str:
    return (
        "You will see a single POST-disaster street-level image.\n"
        "TASKS:\n"
        "1) Predict overall damage level as one of: mild, moderate, severe; also provide confidence 0-100.\n"
        "2) Provide 0-100 integer scores for visible indicators ONLY:\n"
        "   - fallen_trees\n"
        "   - building_debris (scattered materials, collapsed fragments)\n"
        "   - damaged_infrastructure (poles, powerlines, signage, road barriers)\n"
        "   - flooded_roads\n"
        "3) Produce THREE English texts based ONLY on what is visible:\n"
        f"   a) headline: <= {HEADLINE_MAX_WORDS} words, FRONT-LOAD the key indicators.\n"
        f"   b) short_caption: <= {SHORT_CAPTION_MAX_WORDS} words, one sentence, still front-loaded.\n"
        f"   c) long_caption: {LONG_CAPTION_MIN_WORDS}-{LONG_CAPTION_MAX_WORDS} words; front-load most critical indicators.\n\n"
        "STRICT RESPONSE FORMAT (return JSON only):\n"
        "{\n"
        '  "prediction": { "label": "mild|moderate|severe", "confidence": 0-100 },\n'
        '  "indicators": {\n'
        '    "fallen_trees": 0-100,\n'
        '    "building_debris": 0-100,\n'
        '    "damaged_infrastructure": 0-100,\n'
        '    "flooded_roads": 0-100\n'
        "  },\n"
        '  "headline": "...",\n'
        '  "short_caption": "...",\n'
        '  "long_caption": "..."\n'
        "}\n"
    )

# --- 4. Helper Functions ---
def is_image_file(name: str) -> bool:
    return name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))

def clamp_int(x, lo=0, hi=100) -> int:
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except:
        return 0

def normalize_result(d: dict) -> dict:
    """Clean JSON data returned by API"""
    pred = d.get("prediction", {})
    label = str(pred.get("label", "")).strip().lower()
    if label not in {"mild", "moderate", "severe"}: label = "mild"
    
    inds = d.get("indicators", {})
    
    return {
        "pred_label": label,
        "pred_confidence": clamp_int(pred.get("confidence", 0)),
        "fallen_trees": clamp_int(inds.get("fallen_trees", 0)),
        "building_debris": clamp_int(inds.get("building_debris", 0)),
        "damaged_infrastructure": clamp_int(inds.get("damaged_infrastructure", 0)),
        "flooded_roads": clamp_int(inds.get("flooded_roads", 0)),
        "headline": " ".join(str(d.get("headline", "")).split()),
        "short_caption": " ".join(str(d.get("short_caption", "")).split()),
        "long_caption": " ".join(str(d.get("long_caption", "")).split()),
    }

def generate_with_gemini(image_path: str) -> dict:
    prompt = build_prompt()
    image = Image.open(image_path)
    
    # Call API
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image],
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            response_mime_type="application/json", # Force JSON
            system_instruction=SYSTEM_INSTRUCTION
        )
    )
    return json.loads(response.text)

# --- 5. Main Pipeline ---
def run_pipeline():
    print(f"🚀 Starting processing... Target directory: {BASE_DIR}")
    
    # Check path
    if not os.path.exists(BASE_DIR):
        print(f"❌ Error: Folder not found {BASE_DIR}")
        return

    out_csv = os.path.join(BASE_DIR, CSV_NAME)
    fieldnames = [
        "image_path","folder_tag",
        "pred_label","pred_confidence",
        "fallen_trees","building_debris","damaged_infrastructure","flooded_roads",
        "headline","short_caption","long_caption",
    ]
    rows = []

    # Clear old logs
    if os.path.exists(ERR_LOG):
        try: os.remove(ERR_LOG)
        except: pass

    total_files = 0
    
    # Iterate through folders
    for folder_name, weak_label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Skipping non-existent subfolder: {folder_name}")
            continue

        # Collect image tasks
        img_tasks = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if is_image_file(f):
                    img_tasks.append(os.path.join(root, f))
        
        if not img_tasks:
            print(f"⚠️ {folder_name} is empty, no images found")
            continue

        total_files += len(img_tasks)
        
        # Progress bar processing
        print(f"📂 Processing {folder_name} ({len(img_tasks)} images)...")
        for img_path in tqdm(img_tasks):
            try:
                result = generate_with_gemini(img_path)
                norm = normalize_result(result)
            except Exception as e:
                # Log error
                with open(ERR_LOG, "a", encoding="utf-8") as ef:
                    ef.write(f"FILE: {img_path}\nERROR: {str(e)}\n{traceback.format_exc()}\n{'-'*60}\n")
                
                # Fill empty rows
                norm = {k: "" for k in fieldnames if k not in ["image_path", "folder_tag"]}
                norm["headline"] = f"[ERROR] {str(e)}"
            
            rows.append({"image_path": img_path, "folder_tag": weak_label, **norm})

    if total_files == 0:
        print("❌ No images found, please check folder structure.")
        return

    # Save results
    try:
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + "="*50)
        print(f"✅ All done!")
        print(f"📄 Results saved to: {out_csv}")
        print(f"📊 Total images processed: {len(rows)}")
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

if __name__ == "__main__":
    run_pipeline()
