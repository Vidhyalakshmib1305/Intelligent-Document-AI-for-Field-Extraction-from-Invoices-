#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
IDFC HACKATHON - INVOICE FIELD EXTRACTION (EXECUTABLE)
═══════════════════════════════════════════════════════════════════

Command-line executable for hackathon evaluation

Usage: python executable.py <image.png>
Output: output.json

Constraints Met:
✓ NO internet connectivity (offline execution)
✓ GPU: 16GB VRAM or less (tested on T4)
✓ Single PNG input via command line argument
✓ Output: output.json (not results.json)
✓ Processing time: <30s per document

Author: B Vidhyalakshmi
Date: January 2026
"""

import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Deep learning & vision imports
# import torch
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from paddleocr import PaddleOCR
# import cv2
# import numpy as np

# Deep learning & vision imports
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Disable PaddleX before importing PaddleOCR
import os
os.environ['DISABLE_PADDLEX'] = '1'

from paddleocr import PaddleOCR
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 1280  # Prevent OOM on large images

# Global models (loaded once at startup)
model = None
processor = None
ocr = None

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def load_models():
    """Load all AI models once at startup"""
    global model, processor, ocr
    
    if model is None:
        print("🤖 Loading Qwen2-VL-2B model...", file=sys.stderr)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        print("   ✓ VLM loaded", file=sys.stderr)
    
    if ocr is None:
        print("📝 Loading PaddleOCR...", file=sys.stderr)
        ocr = PaddleOCR(
    use_textline_orientation=True,
    lang='en'
)

        print("   ✓ OCR loaded", file=sys.stderr)

# ═══════════════════════════════════════════════════════════════════
# SIGNATURE & STAMP DETECTION (Computer Vision + VLM Hybrid)
# ═══════════════════════════════════════════════════════════════════
def detect_signatures_stamps(image_path: str, image_pil: Image.Image) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect signatures and stamps using hybrid approach:
    - Primary: Computer Vision (OpenCV)
    - Fallback: Vision-Language Model (Qwen2-VL)
    
    Returns:
        signatures: List of {"present": bool, "bbox": [x1,y1,x2,y2]}
        stamps: List of {"present": bool, "bbox": [x1,y1,x2,y2]}
    """
    try:
        # Load image for CV processing
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise ValueError("Could not load image with OpenCV")
            
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        signatures = []
        stamps = []
        
        # ═══════════════════════════════════════════════════════════
        # METHOD 1: STAMP DETECTION (Circular/Rectangular Dense Regions)
        # ═══════════════════════════════════════════════════════════
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (stamps are typically 500-20,000 pixels)
            if 500 < area < 20000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Stamps are nearly square/circular (aspect ratio ~1)
                if 0.6 < aspect_ratio < 1.4:
                    roi = gray[y:y+h, x:x+w]
                    density = np.sum(roi < 200) / (w * h) if w * h > 0 else 0
                    
                    # Check density (stamps have dense ink)
                    if density > 0.3:  # 30% dark pixels
                        stamps.append({
                            "present": True,
                            "bbox": [int(x), int(y), int(x+w), int(y+h)]
                        })
        
        # ═══════════════════════════════════════════════════════════
        # METHOD 2: SIGNATURE DETECTION (Cursive Strokes)
        # ═══════════════════════════════════════════════════════════
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        for i in range(1, num_labels):  # Skip background (0)
            x, y, w, h, area = stats[i]
            
            # Signatures: medium size, horizontal aspect, in bottom half
            if 1000 < area < 15000 and w > h * 1.5 and y > height * 0.4:
                signatures.append({
                    "present": True,
                    "bbox": [int(x), int(y), int(x+w), int(y+h)]
                })
        
        # ═══════════════════════════════════════════════════════════
        # METHOD 3: VLM FALLBACK (if CV found nothing)
        # ═══════════════════════════════════════════════════════════
        if not signatures and not stamps:
            vlm_sig, vlm_stamp = detect_with_vlm(image_pil)
            if vlm_sig['present']:
                signatures.append(vlm_sig)
            if vlm_stamp['present']:
                stamps.append(vlm_stamp)
        
        # Return at least one entry (even if not detected)
        if not signatures:
            signatures = [{"present": False, "bbox": [0, 0, 0, 0]}]
        if not stamps:
            stamps = [{"present": False, "bbox": [0, 0, 0, 0]}]
        
        return signatures, stamps
        
    except Exception as e:
        # Safe fallback on any error
        print(f"   ⚠️  CV detection failed: {e}", file=sys.stderr)
        return [{"present": False, "bbox": [0, 0, 0, 0]}], [{"present": False, "bbox": [0, 0, 0, 0]}]

def detect_with_vlm(image_pil: Image.Image) -> Tuple[Dict, Dict]:
    """VLM fallback for signature/stamp detection"""
    try:
        prompt = """Look at this invoice carefully. Detect:
1. Handwritten signature (cursive writing, NOT printed text)
2. Company stamp (circular or rectangular ink stamp)

Return ONLY JSON (no markdown, no explanation):
{"signature": {"present": true/false, "bbox": [x1,y1,x2,y2]},
 "stamp": {"present": true/false, "bbox": [x1,y1,x2,y2]}}

If not found, use {"present": false, "bbox": [0,0,0,0]}"""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_pil},
                {"type": "text", "text": prompt}
            ]
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON from response
        json_start = output_text.find('{')
        json_end = output_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(output_text[json_start:json_end])
            return result.get('signature', {"present": False, "bbox": [0,0,0,0]}), \
                   result.get('stamp', {"present": False, "bbox": [0,0,0,0]})
    except Exception as e:
        print(f"   ⚠️  VLM detection failed: {e}", file=sys.stderr)
    
    return {"present": False, "bbox": [0,0,0,0]}, {"present": False, "bbox": [0,0,0,0]}

# ═══════════════════════════════════════════════════════════════════
# OCR & TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def run_ocr(image_path: str) -> str:
    """Extract all text from image using PaddleOCR"""
    try:
        result = ocr.ocr(image_path, cls=True)
        if result and result[0]:
            text_lines = [line[1][0] for line in result[0]]
            return " ".join(text_lines)
        return ""
    except Exception as e:
        print(f"   ⚠️  OCR failed: {e}", file=sys.stderr)
        return ""

def extract_hp_from_text(text: str) -> int:
    """
    Extract horse power using multiple regex patterns
    Handles English, Hindi, and Gujarati text
    """
    patterns = [
        r'(\d+)\s*HP\b',
        r'(\d+)\s*H\.P\.',
        r'\b(\d+)\s*हॉर्स',  # Hindi: हॉर्सपावर
        r'Horse\s*Power[:\s]*(\d+)',
        r'HP[:\s]*(\d+)',
        r'Power[:\s]*(\d+)\s*HP',
        r'(\d+)\s*એચપી',  # Gujarati: એચપી
        r'(\d+)\s*hp\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            hp = int(match)
            # Validate: tractors typically 20-150 HP
            if 20 <= hp <= 150:
                return hp
    
    return 0

def extract_cost_from_text(text: str) -> int:
    """
    Extract FINAL asset cost (including GST/taxes)
    Uses 3-tier priority system for robustness
    """
    
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 1: Explicit Total Keywords
    # ═══════════════════════════════════════════════════════════
    total_patterns = [
        r'(?:Grand\s*Total|Total\s*Amount|Net\s*Amount|Final\s*Amount)[:\s]*(?:Rs\.?|₹)?\s*([\d,]+)',
        r'(?:Total|Amount)\s*(?:Payable|Due)[:\s]*(?:Rs\.?|₹)?\s*([\d,]+)',
        r'(?:with|including)\s*(?:GST|Tax|Taxes)[:\s]*(?:Rs\.?|₹)?\s*([\d,]+)',
        r'कुल\s*राशि[:\s]*(?:₹)?\s*([\d,]+)',  # Hindi: कुल राशि
        r'કુલ\s*રકમ[:\s]*(?:₹)?\s*([\d,]+)',  # Gujarati: કુલ રકમ
    ]
    
    for pattern in total_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            clean_num = match.replace(',', '')
            if clean_num.isdigit():
                value = int(clean_num)
                # Validate: tractor cost range 3L-30L
                if 300000 <= value <= 3000000:
                    return value
    
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 2: Contextual Search (near "total" keyword)
    # ═══════════════════════════════════════════════════════════
    total_idx = text.lower().find('total')
    if total_idx > 0:
        # Extract text ±100 chars around "total"
        total_section = text[max(0, total_idx - 100):min(len(text), total_idx + 200)]
        numbers = re.findall(r'[\d,]+', total_section)
        costs = []
        for num in numbers:
            clean_num = num.replace(',', '')
            if clean_num.isdigit() and len(clean_num) >= 5:
                value = int(clean_num)
                if 300000 <= value <= 3000000:
                    costs.append(value)
        if costs:
            return max(costs)  # Return highest value in section
    
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 3: Heuristic (all large numbers)
    # ═══════════════════════════════════════════════════════════
    all_numbers = re.findall(r'[\d,]+', text)
    costs = []
    for num in all_numbers:
        clean_num = num.replace(',', '')
        if clean_num.isdigit() and len(clean_num) >= 6:  # At least 6 digits
            value = int(clean_num)
            if 300000 <= value <= 3000000:
                costs.append(value)
    
    return max(costs) if costs else 0

# ═══════════════════════════════════════════════════════════════════
# VLM FIELD EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_fields_vlm(image_pil: Image.Image, ocr_text: str) -> Dict:
    """
    Use Qwen2-VL to extract text fields (dealer, model, HP, cost)
    OCR text provides context to improve accuracy
    """
    
    prompt = f"""Analyze this tractor invoice/quotation and extract key fields.

OCR Text Reference (for context): {ocr_text[:800]}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "dealer_name": "full dealer/company name (keep vernacular text exactly as-is)",
  "model_name": "exact tractor model name",
  "horse_power": 0,
  "asset_cost": 0,
  "confidence": 0.85
}}

CRITICAL EXTRACTION RULES:
1. dealer_name: Company/dealer name - preserve Hindi/Gujarati/Marathi characters EXACTLY
2. model_name: Full model name like "SWARAJ 744 FE", "Mahindra 575 DI", "John Deere 5050D"
3. horse_power: ONLY numeric value (e.g., 50 not "50 HP")
4. asset_cost: FINAL total cost INCLUDING GST/taxes (NOT base/ex-showroom price)
5. confidence: 0.0-1.0 based on field clarity

If field not found or unclear, use empty string "" or 0"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Parse JSON from VLM response
    json_start = output_text.find('{')
    json_end = output_text.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_str = output_text[json_start:json_end]
        # Clean up common JSON issues
        json_str = json_str.replace("'", '"').replace('True', 'true').replace('False', 'false')
        return json.loads(json_str)
    
    raise ValueError("No valid JSON in VLM response")

# ═══════════════════════════════════════════════════════════════════
# MAIN EXTRACTION PIPELINE
# ═══════════════════════════════════════════════════════════════════
def extract_invoice_fields(image_path: str) -> Dict:
    """
    Complete extraction pipeline for a single invoice
    
    Pipeline:
    1. Load & preprocess image
    2. OCR text extraction
    3. Signature/stamp detection (CV + VLM)
    4. Field extraction (VLM)
    5. Post-processing & validation
    6. Confidence scoring
    """
    
    start_time = time.time()
    
    try:
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Load and Resize Image
        # ═══════════════════════════════════════════════════════════
        image_pil = Image.open(image_path).convert('RGB')
        
        # Resize large images to prevent OOM
        if max(image_pil.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image_pil.size)
            new_size = tuple(int(dim * ratio) for dim in image_pil.size)
            image_pil = image_pil.resize(new_size, Image.LANCZOS)
            print(f"   ℹ️  Resized to {new_size}", file=sys.stderr)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: OCR Text Extraction
        # ═══════════════════════════════════════════════════════════
        print("   🔤 Running OCR...", file=sys.stderr)
        ocr_text = run_ocr(image_path)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Signature & Stamp Detection
        # ═══════════════════════════════════════════════════════════
        print("   ✍️  Detecting signatures/stamps...", file=sys.stderr)
        signatures, stamps = detect_signatures_stamps(image_path, image_pil)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: VLM Field Extraction
        # ═══════════════════════════════════════════════════════════
        print("   🤖 Extracting fields with VLM...", file=sys.stderr)
        vlm_fields = extract_fields_vlm(image_pil, ocr_text)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: Post-Processing (Regex Fallbacks)
        # ═══════════════════════════════════════════════════════════
        # Fix HP if VLM missed it
        if vlm_fields.get('horse_power', 0) == 0:
            hp_from_ocr = extract_hp_from_text(ocr_text)
            if hp_from_ocr > 0:
                vlm_fields['horse_power'] = hp_from_ocr
                print(f"   ✓ HP recovered from OCR: {hp_from_ocr}", file=sys.stderr)
        
        # Fix cost if VLM missed it
        if vlm_fields.get('asset_cost', 0) == 0:
            cost_from_ocr = extract_cost_from_text(ocr_text)
            if cost_from_ocr > 0:
                vlm_fields['asset_cost'] = cost_from_ocr
                print(f"   ✓ Cost recovered from OCR: ₹{cost_from_ocr:,}", file=sys.stderr)
        
        # ═══════════════════════════════════════════════════════════
        # STEP 6: Confidence Scoring
        # ═══════════════════════════════════════════════════════════
        base_confidence = vlm_fields.get('confidence', 0.5)
        
        # Count successfully extracted fields
        fields_found = sum([
            bool(vlm_fields.get('dealer_name')),
            bool(vlm_fields.get('model_name')),
            vlm_fields.get('horse_power', 0) > 0,
            vlm_fields.get('asset_cost', 0) > 0,
            any(s['present'] for s in signatures),
            any(s['present'] for s in stamps)
        ])
        
        # Adjust confidence based on completeness
        adjusted_confidence = base_confidence * (fields_found / 6.0) * 1.15
        adjusted_confidence = min(adjusted_confidence, 0.99)  # Cap at 0.99
        
        processing_time = time.time() - start_time
        
        # ═══════════════════════════════════════════════════════════
        # Build Final Output
        # ═══════════════════════════════════════════════════════════
        result = {
            "doc_id": Path(image_path).stem,
            "fields": {
                "dealer_name": vlm_fields.get('dealer_name', ''),
                "model_name": vlm_fields.get('model_name', ''),
                "horse_power": vlm_fields.get('horse_power', 0),
                "asset_cost": vlm_fields.get('asset_cost', 0),
                "signature": signatures[0] if len(signatures) == 1 else signatures,
                "stamp": stamps[0] if len(stamps) == 1 else stamps
            },
            "confidence": round(adjusted_confidence, 3),
            "processing_time_sec": round(processing_time, 2),
            "cost_estimate_usd": 0.0  # Open-source = $0
        }
        
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ❌ Extraction failed: {e}", file=sys.stderr)
        return {
            "doc_id": Path(image_path).stem,
            "error": str(e)[:200],
            "processing_time_sec": round(processing_time, 2),
            "cost_estimate_usd": 0.0
        }

# ═══════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════
def main():
    """Command-line interface for hackathon evaluation"""
    
    # Validate arguments
    if len(sys.argv) != 2:
        print("Usage: python executable.py <image.png>", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  python executable.py invoice_001.png", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if file is PNG
    if not image_path.lower().endswith('.png'):
        print(f"⚠️  Warning: Expected PNG file, got {Path(image_path).suffix}", file=sys.stderr)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"IDFC Invoice Field Extraction", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Input: {image_path}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    
    # Load models (only once)
    load_models()
    
    # Extract fields from invoice
    print(f"🚀 Processing invoice...\n", file=sys.stderr)
    result = extract_invoice_fields(image_path)
    
    # Write output to output.json
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"✅ Extraction Complete", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: output.json", file=sys.stderr)
    print(f"Processing time: {result.get('processing_time_sec', 0)}s", file=sys.stderr)
    
    if 'error' not in result:
        print(f"Confidence: {result.get('confidence', 0):.2%}", file=sys.stderr)
        print(f"Fields extracted: {sum([bool(v) for v in result.get('fields', {}).values()])}/6", file=sys.stderr)
    else:
        print(f"Status: FAILED - {result.get('error', 'Unknown error')}", file=sys.stderr)
    
    print(f"{'='*60}\n", file=sys.stderr)

if __name__ == "__main__":
    main()
