import os
import json
import logging
import io
from datetime import datetime
import fitz  # PyMuPDF
import concurrent.futures
from google.cloud import vision
from google.oauth2 import service_account

# --- Configuration ---
PDF_PATH = r"C:\Users\rcgop\Downloads\The Unfair Advantage-20250927T082820Z-1-001\The Unfair Advantage\Business Plans\sample_odia_1.pdf"  # Update PDF path
SERVICE_ACCOUNT_JSON = r"C:\Users\swati\OneDrive\Desktop\text_json\gen-lang-client-0858700453-3ffbc840e488[1].json"  # Your service account JSON
OUTPUT_DIR = "pdf_output_threaded"  # Directory to save results
IMAGE_DPI = 300              # DPI for PDF to image conversion
CLEAN_IMAGES = True          # Delete images after OCR

# --- NEW: Threading Configuration ---
# Number of parallel API requests to make.
# Increase this for faster processing, but be mindful of API rate limits. 10 is a good start.
MAX_WORKERS = 10

# Language configuration
LANGUAGE_HINTS = ["en", "hi", "or", "bn", "ta", "te", "ml", "kn", "gu", "pa", "mr", "as", "es", "fr", "de", "ja", "ko", "zh", "ar", "ru"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Vision client
try:
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
    client = vision.ImageAnnotatorClient(credentials=credentials)
except Exception as e:
    logging.error(f"Failed to initialize Google Cloud client: {e}")
    exit()

# ---------------- Functions ---------------- #

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list[str]:
    """Convert each page of a PDF into a PNG image."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Converting {doc.page_count} pages of '{pdf_path}' to images...")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image_filename = os.path.join(output_dir, f"page_{page_num + 1}.png")
            pix.save(image_filename)
            image_paths.append(image_filename)
        doc.close()
        logging.info(f"Successfully converted PDF to {len(image_paths)} images.")
    except Exception as e:
        logging.error(f"Error during PDF to image conversion: {e}")
        return []
    return image_paths

def perform_ocr_on_image(image_path: str) -> vision.AnnotateImageResponse:
    """Perform OCR on a single image file."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    image_context = vision.ImageContext(language_hints=LANGUAGE_HINTS)
    return client.document_text_detection(image=image, image_context=image_context)

def process_single_page(image_path: str, page_num: int) -> dict:
    """
    Worker function: Performs OCR on one image, processes the result, and returns structured data.
    This function will be run in parallel by multiple threads.
    """
    logging.info(f"[Thread] Processing page {page_num} ({os.path.basename(image_path)})...")
    try:
        response = perform_ocr_on_image(image_path)
        
        # Process the raw response
        annotation = response.full_text_annotation
        full_text = annotation.text if annotation else ""
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        detected_languages = detect_languages_in_text(full_text)
        
        page_data = {
            "page_number": page_num,
            "full_text": '\n'.join(lines),
            "word_count": len(' '.join(lines).split()) if lines else 0,
            "has_content": bool(lines),
            "detected_languages": detected_languages,
        }
        logging.info(f"[Thread] Finished page {page_num}. Words: {page_data['word_count']}")
        return page_data
    except Exception as e:
        logging.error(f"[Thread] FAILED to process page {page_num}: {e}")
        return {"page_number": page_num, "error": str(e), "has_content": False}
    finally:
        # Clean up the image file immediately after processing
        if CLEAN_IMAGES and os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"[Thread] Cleaned up image: {os.path.basename(image_path)}")

def detect_languages_in_text(text: str) -> list[str]:
    """Simple language detection based on character patterns."""
    detected_langs = []
    if any('\u0B00' <= char <= '\u0B7F' for char in text): detected_langs.append('Odia')
    if any(char.isascii() and char.isalpha() for char in text): detected_langs.append('Latin_Script')
    return detected_langs if detected_langs else ['Unknown']

def save_results(all_pages_data: list, output_dir: str, unique_filename: str):
    """Saves the final aggregated results to JSON and TXT files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save complete JSON
    json_path = os.path.join(output_dir, f"{unique_filename}_complete.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved complete JSON to: {json_path}")
    
    # Save plain text file
    all_text = [f"=== Page {p['page_number']} ===\n{p['full_text']}" for p in all_pages_data if not p.get('error')]
    txt_path = os.path.join(output_dir, f"{unique_filename}_extracted_text.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_text))
    logging.info(f"Saved plain text to: {txt_path}")

# ---------------- Main ---------------- #

def main():
    if not os.path.exists(PDF_PATH):
        logging.error(f"PDF file not found at '{PDF_PATH}'.")
        return

    unique_filename = os.path.splitext(os.path.basename(PDF_PATH))[0]
    
    # Step 1: Convert PDF to images (this is still sequential)
    image_files = convert_pdf_to_images(PDF_PATH, OUTPUT_DIR)
    if not image_files:
        logging.error("No images were generated from the PDF. Exiting.")
        return

    all_pages_data = []
    logging.info(f"Starting parallel OCR processing with {MAX_WORKERS} workers...")

    # Step 2: Process images in parallel using a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each page processing task
        future_to_page = {
            executor.submit(process_single_page, image_path, i + 1): i + 1
            for i, image_path in enumerate(image_files)
        }
        
        # As each future completes, collect its result
        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                page_data = future.result()
                all_pages_data.append(page_data)
            except Exception as exc:
                logging.error(f'Page {page_num} generated an exception: {exc}')

    # Step 3: Sort results by page number, as threads can finish out of order
    all_pages_data.sort(key=lambda p: p['page_number'])

    # Step 4: Save the aggregated and sorted results
    logging.info("All pages processed. Saving final results...")
    save_results(all_pages_data, OUTPUT_DIR, unique_filename)

    logging.info("=" * 50)
    logging.info("PARALLEL OCR PROCESSING COMPLETE")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()