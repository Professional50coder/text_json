import fitz  # PyMuPDF
import io
import os
import json
import logging
from datetime import datetime
from google.cloud import vision
from google.oauth2 import service_account

# --- Configuration ---
PDF_PATH = r"C:\Users\USER\Downloads\sample_english_5.pdf"  # Update PDF path
OUTPUT_DIR = "pdf_output"  # Directory to save images and final JSON
IMAGE_DPI = 300            # DPI for PDF to image conversion
CLEAN_IMAGES = True        # Delete images after OCR
SERVICE_ACCOUNT_JSON = r"gen-lang-client-0858700453-3f96694fab49.json"  # Your service account JSON

# Language configuration for multilingual support
LANGUAGE_HINTS = ["en", "hi", "or", "bn", "ta", "te", "ml", "kn", "gu", "pa", "mr", "as", "es", "fr", "de", "ja", "ko", "zh", "ar", "ru"]  # Added Indian languages including Odia

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Vision client using service account credentials
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
client = vision.ImageAnnotatorClient(credentials=credentials)

# ---------------- Functions ---------------- #

def get_unique_filename(pdf_path: str) -> str:
    """Generate a unique filename based on PDF name and timestamp."""
    # Extract PDF name without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{pdf_name}_{timestamp}"

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
            logging.info(f"Saved {image_filename}")
        doc.close()
    except Exception as e:
        logging.error(f"Error during PDF to image conversion: {e}")
        return []
    return image_paths

def perform_ocr_on_image(client: vision.ImageAnnotatorClient, image_path: str) -> vision.AnnotateImageResponse:
    """Perform OCR on a single image using Google Vision API with multilingual support."""
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Configure image context for multilingual detection
    image_context = vision.ImageContext(language_hints=LANGUAGE_HINTS)
    
    return client.document_text_detection(image=image, image_context=image_context)

def detect_languages_in_text(text: str) -> list[str]:
    """Simple language detection based on character patterns."""
    detected_langs = []
    
    # Check for different script patterns
    if any('\u0900' <= char <= '\u097F' for char in text):  # Devanagari (Hindi, Marathi, Nepali)
        detected_langs.append('Devanagari_Script')
    if any('\u0B00' <= char <= '\u0B7F' for char in text):  # Odia
        detected_langs.append('Odia')
    if any('\u0980' <= char <= '\u09FF' for char in text):  # Bengali/Assamese
        detected_langs.append('Bengali_Assamese')
    if any('\u0B80' <= char <= '\u0BFF' for char in text):  # Tamil
        detected_langs.append('Tamil')
    if any('\u0C00' <= char <= '\u0C7F' for char in text):  # Telugu
        detected_langs.append('Telugu')
    if any('\u0D00' <= char <= '\u0D7F' for char in text):  # Malayalam
        detected_langs.append('Malayalam')
    if any('\u0C80' <= char <= '\u0CFF' for char in text):  # Kannada
        detected_langs.append('Kannada')
    if any('\u0A80' <= char <= '\u0AFF' for char in text):  # Gujarati
        detected_langs.append('Gujarati')
    if any('\u0A00' <= char <= '\u0A7F' for char in text):  # Gurmukhi (Punjabi)
        detected_langs.append('Punjabi')
    if any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese
        detected_langs.append('Chinese')
    if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' for char in text):  # Japanese
        detected_langs.append('Japanese')
    if any('\uAC00' <= char <= '\uD7AF' for char in text):  # Korean
        detected_langs.append('Korean')
    if any('\u0600' <= char <= '\u06FF' for char in text):  # Arabic
        detected_langs.append('Arabic')
    if any('\u0400' <= char <= '\u04FF' for char in text):  # Cyrillic (Russian)
        detected_langs.append('Russian')
    if any(char.isascii() and char.isalpha() for char in text):  # Latin script
        detected_langs.append('Latin_Script')
    
    return detected_langs if detected_langs else ['Unknown']

def process_ocr_response_to_structured_json(response: vision.AnnotateImageResponse, page_num: int) -> dict:
    """Process OCR response into clean text format with language detection."""
    # Get the full text content
    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    
    # Clean up the text - remove excessive whitespace but preserve structure
    lines = []
    if full_text:
        for line in full_text.split('\n'):
            line = line.strip()
            if line:  # Only add non-empty lines
                lines.append(line)
    
    # Detect languages in the text
    detected_languages = detect_languages_in_text(full_text)
    
    # Simple structure focusing on actual content
    page_data = {
        "page_number": page_num,
        "text_content": lines,
        "full_text": '\n'.join(lines),
        "has_content": bool(lines),
        "detected_languages": detected_languages,
    }
    
    # Extract key-value pairs if any text exists
    kv_pairs = {}
    for line in lines:
        if ':' in line and len(line.split(':')) == 2:
            key, value = line.split(':', 1)
            kv_pairs[key.strip()] = value.strip()
    
    if kv_pairs:
        page_data["key_value_pairs"] = kv_pairs

    return page_data

def save_results(all_pages_data: list, output_dir: str, unique_filename: str) -> dict:
    """Save OCR results in multiple formats with unique filenames."""
    results = {}
    
    # # 1. Save complete JSON with all data
    # json_path = os.path.join(output_dir, f"{unique_filename}_complete.json")
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(all_pages_data, f, indent=2, ensure_ascii=False)
    # results['complete_json'] = json_path
    
    # 2. Save text-only version
    text_only_data = []
    all_text = []
    for page in all_pages_data:
        if 'full_text' in page and page['has_content']:
            text_only_data.append({
                "page_number": page['page_number'],
                "text": page['full_text'],
                "languages": page.get('detected_languages', []),
                "word_count": page.get('word_count', 0)
            })
            all_text.append(f"=== Page {page['page_number']} ===\n{page['full_text']}")
    
    text_json_path = os.path.join(output_dir, f"{unique_filename}_text_only.json")
    with open(text_json_path, 'w', encoding='utf-8') as f:
        json.dump(text_only_data, f, indent=2, ensure_ascii=False)
    results['text_json'] = text_json_path
    
    # # 3. Save as plain text file
    # txt_path = os.path.join(output_dir, f"{unique_filename}_extracted_text.txt")
    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     f.write('\n\n'.join(all_text))
    # results['text_file'] = txt_path
    
    # 4. Save summary
    # total_pages = len(all_pages_data)
    # pages_with_content = sum(1 for page in all_pages_data if page.get('has_content', False))
    # total_words = sum(page.get('word_count', 0) for page in all_pages_data)
    # all_languages = set()
    # for page in all_pages_data:
    #     all_languages.update(page.get('detected_languages', []))
    
    # summary = {
    #     "processing_info": {
    #         "timestamp": datetime.now().isoformat(),
    #         "source_pdf": PDF_PATH,
    #         "unique_id": unique_filename
    #     },
    #     "statistics": {
    #         "total_pages": total_pages,
    #         "pages_with_content": pages_with_content,
    #         "total_words": total_words,
    #         "detected_languages": list(all_languages)
    #     },
    #     "files_created": results
    # }
    
    # summary_path = os.path.join(output_dir, f"{unique_filename}_summary.json")
    # with open(summary_path, 'w', encoding='utf-8') as f:
    #     json.dump(summary, f, indent=2, ensure_ascii=False)
    # results['summary'] = summary_path
    
    return results

# ---------------- Main ---------------- #

def main():
    if not os.path.exists(PDF_PATH):
        logging.error(f"PDF file not found at '{PDF_PATH}'. Update PDF_PATH.")
        return

    # Generate unique filename for this processing run
    unique_filename = get_unique_filename(PDF_PATH)
    logging.info(f"Processing PDF with unique ID: {unique_filename}")

    # Convert PDF to images
    image_files = convert_pdf_to_images(PDF_PATH, OUTPUT_DIR, IMAGE_DPI)
    if not image_files:
        logging.error("No images were generated. Exiting.")
        return

    all_pages_data = []

    for i, image_path in enumerate(image_files):
        logging.info(f"Performing OCR on {image_path}...")
        try:
            response = perform_ocr_on_image(client, image_path)
            page_data = process_ocr_response_to_structured_json(response, i + 1)
            all_pages_data.append(page_data)
            
            # Log processing results
            languages = ', '.join(page_data.get('detected_languages', ['None']))
            logging.info(f"Page {i + 1} complete - Characters: {page_data.get('character_count', 0)}, "
                        f"Words: {page_data.get('word_count', 0)}, Languages: {languages}")

            if CLEAN_IMAGES:
                os.remove(image_path)
                logging.info(f"Removed image file: {image_path}")
        except Exception as e:
            logging.error(f"OCR failed for {image_path}: {e}")
            all_pages_data.append({
                "page_number": i + 1, 
                "error": str(e),
                "has_content": False,
                "processing_timestamp": datetime.now().isoformat()
            })

    # Save results in multiple formats
    logging.info("Saving results...")
    saved_files = save_results(all_pages_data, OUTPUT_DIR, unique_filename)
    
    # Final summary
    logging.info("=" * 50)
    logging.info("OCR PROCESSING COMPLETE")
    logging.info("=" * 50)
    logging.info(f"Unique ID: {unique_filename}")
    logging.info(f"Total pages processed: {len(all_pages_data)}")
    
    # Log all created files
    for file_type, file_path in saved_files.items():
        logging.info(f"{file_type.upper()}: {file_path}")
    
    logging.info("=" * 50)

if __name__ == "__main__":
    main()