from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
from full_multi_updated2 import get_unique_filename, convert_pdf_to_images, perform_ocr_on_image, process_ocr_response_to_structured_json, save_results, client, IMAGE_DPI, OUTPUT_DIR, CLEAN_IMAGES


class PDFProcessRequest(BaseModel):
    pdf_path: str = None
    pdf_link: str = None

router = APIRouter()

@router.post("/process-pdf")
async def process_pdf_api(payload: PDFProcessRequest):
    pdf_path = payload.pdf_path
    pdf_link = payload.pdf_link

    # Use pdf_path if provided, else pdf_link
    if pdf_path and os.path.exists(pdf_path):
        pdf_to_use = pdf_path
    elif pdf_link:
        # Download PDF from link
        import requests
        response = requests.get(pdf_link)
        if response.status_code == 200:
            temp_pdf_path = os.path.join(OUTPUT_DIR, "temp_input.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(response.content)
            pdf_to_use = temp_pdf_path
        else:
            raise HTTPException(status_code=400, detail="Failed to download PDF from link.")
    else:
        raise HTTPException(status_code=400, detail="No valid PDF path or link provided.")

    # Run the OCR pipeline
    try:
        unique_filename = get_unique_filename(pdf_to_use)
        image_files = convert_pdf_to_images(pdf_to_use, OUTPUT_DIR, IMAGE_DPI)
        all_pages_data = []
        for i, image_path in enumerate(image_files):
            response = perform_ocr_on_image(client, image_path)
            page_data = process_ocr_response_to_structured_json(response, i + 1)
            all_pages_data.append(page_data)
            if CLEAN_IMAGES:
                os.remove(image_path)
        saved_files = save_results(all_pages_data, OUTPUT_DIR, unique_filename)
        # Return the main JSON result
        return JSONResponse(content=all_pages_data)
    except Exception as e:
        logging.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp PDF if downloaded
        if pdf_link and os.path.exists(os.path.join(OUTPUT_DIR, "temp_input.pdf")):
            os.remove(os.path.join(OUTPUT_DIR, "temp_input.pdf"))
