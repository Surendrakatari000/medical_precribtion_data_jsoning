import os
import cv2
import json
import re
from paddleocr import PaddleOCR
import re

# 1. Environment configuration
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_use_onednn'] = '0' 

# 2. Initialize OCR with higher unclip ratio for handwriting
ocr = PaddleOCR(
    use_textline_orientation=True, 
    lang='en', 
    enable_mkldnn=False,
    text_det_unclip_ratio=2.2, 
    text_det_box_thresh=0.0
)

def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    
    # Using predict (standardized for PaddleX/PaddleOCR newer versions)
    result = ocr.predict(img)
    
    # Access the 'rec_texts' key which contains your list of strings
    output_dict = result[0] 
    if 'rec_texts' in output_dict:
        text_lines = output_dict['rec_texts']
        return "\n".join(text_lines)
    return ""



def clean_and_parse(raw_text):
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    
    data = {
        "patient_name": "Not Found",
        "doctor_name": "Not Found",
        "medicines": []
    }
    
    name_buffer = []
    # Keywords that should never be part of a medicine name
    ignore_keywords = ["AGE", "DATE", "MBBS", "HOSPITAL", "CITY", "SEX", "GENDER", "RX", "HISTORY", "PT", "NME", "NAME"]
    used_indices = set()

    for i, line in enumerate(lines):
        if i in used_indices: continue
        line_upper = line.upper()

        # 1. Improved Patient Name Extraction (Handles 'Pt', 'Nme:', 'Lakshmi' on separate lines)
        if any(x == line_upper for x in ["PT", "NAME", "PATIENT", "P NME"]):
            # Look ahead up to 2 lines to find the actual name
            for offset in range(1, 3):
                if i + offset < len(lines):
                    candidate = lines[i + offset]
                    # If the next line is just "Nme:", skip it and look at the one after
                    if candidate.upper() in ["NME:", "NAME:", "NME"]:
                        used_indices.add(i + offset)
                        continue
                    # Capture the name and stop looking
                    data["patient_name"] = re.sub(r'[^a-zA-Z\s]', '', candidate).strip()
                    used_indices.add(i + offset)
                    break
            continue

        # 2. Doctor Name Extraction
        if any(x in line_upper for x in ["DR.", "DR:", "DOCTOR"]):
            val = line.split(':')[-1].strip() if ':' in line else line
            data["doctor_name"] = val.replace("MBBS", "").replace("Dr.", "").replace("Dr:", "").strip()
            continue

        # 3. Frequency Detection & Buffer Processing
        freq_match = re.search(r'(\d\s*-\s*\d\s*-\s*\d)', line)
        if freq_match:
            # Filter the buffer to remove fragments like 'Tab', 'Pt', or empty strings
            cleaned_buffer = []
            for item in name_buffer:
                item_up = item.upper()
                # Skip lines that are just labels or medical prefixes
                if item_up in ignore_keywords or item_up in ["TAB", "CAP", "T", "C"]:
                    continue
                cleaned_buffer.append(item)

            full_text = " ".join(cleaned_buffer).strip()
            
            # Regex for dosage: looks for numbers + units
            dosage_match = re.search(r'(\d+)\s*(mg|g|ml|mcg|tab|cap|my|moy|0og)?', full_text, re.IGNORECASE)
            
            dosage_str = ""
            med_name = full_text
            
            if dosage_match:
                dosage_str = dosage_match.group(0)
                clean_dosage = dosage_str.lower().replace('0og', 'mg').replace('moy', 'mg').replace('my', 'mg')
                med_name = full_text.replace(dosage_str, "").strip()
            else:
                clean_dosage = ""

            # Clean up medicine name (strip 'Tab', 'Cap', etc. from the start)
            med_name = re.sub(r'^(Tab|Cap|T\s|C\s|Rx)\s*', '', med_name, flags=re.IGNORECASE).strip()
            
            if med_name:
                data["medicines"].append({
                    "name": med_name.strip(','),
                    "dosage": clean_dosage,
                    "frequency": freq_match.group(1).replace(" ", "")
                })
            
            name_buffer = []
            continue

        # 4. Buffer logic: Only add if not already used or a header
        if line_upper not in ignore_keywords and line_upper not in ["TAB", "CAP"]:
             # Check if this line was already used for Patient/Doctor name
             if line != data["patient_name"] and line != data["doctor_name"]:
                name_buffer.append(line)

    return data

if __name__ == "__main__":
    image_path = "./second.png"
    try:
        raw = extract_text(image_path)
        print("--- RAW OCR OUTPUT ---")
        print(raw)
        
        final_json = clean_and_parse(raw)
        
        print("\n--- FINAL JSON OUTPUT ---")
        print(json.dumps(final_json, indent=4))
        
        with open("final_output.json", "w") as f:
            json.dump(final_json, f, indent=4)
            
    except Exception as e:
        print(f"Error: {e}")
