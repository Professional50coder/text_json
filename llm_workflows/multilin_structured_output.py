from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import logging

import os
import json
from llm_workflows.google_translate_json import GoogleTranslateJSONConverter

def call_google_translate_json(structured_output: Any, language: str) -> dict:
    # Parse structured_output from JSON string if needed
    if isinstance(structured_output, str):
        try:
            structured_output = json.loads(structured_output)
        except Exception:
            pass  # If not valid JSON, keep as string
    translator = GoogleTranslateJSONConverter()
    return translator.translate_json_content(structured_output, language)

router = APIRouter()


class TranslationPayload(BaseModel):
    structured_output: Any
    language: str

class FeedbackTranslationPayload(BaseModel):
    feedbacks: Any
    language: str


@router.post("/translate-structured-output")
async def translate_structured_output(payload: TranslationPayload):
    try:
        result = call_google_translate_json(payload.structured_output, payload.language)
        return {"translated_output": result, "language": payload.language}
    except Exception as e:
        logging.error(f"Translation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate-feedbacks")
async def translate_feedbacks(payload: FeedbackTranslationPayload):
    try:
        feedbacks = payload.feedbacks
        if isinstance(feedbacks, str):
            try:
                feedbacks = json.loads(feedbacks)
            except Exception:
                pass
        translator = GoogleTranslateJSONConverter()
        result = translator.translate_json_content(feedbacks, payload.language)
        return {"translated_feedbacks": result, "language": payload.language}
    except Exception as e:
        logging.error(f"Feedback Translation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
