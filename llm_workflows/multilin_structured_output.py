from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import logging

import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI

def call_llm_translate(structured_output: Any, language: str) -> str:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return "Google API key is required. Please set GOOGLE_API_KEY in your .env file."
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=512,
            timeout=30,
            max_retries=2,
            google_api_key=google_api_key,
        )
    except Exception as e:
        return f"Error initializing LLM: {str(e)}"

    # Parse structured_output from JSON string if needed
    if isinstance(structured_output, str):
        try:
            structured_output = json.loads(structured_output)
        except Exception:
            pass  # If not valid JSON, keep as string

    prompt = (
        f"You are a professional translator. Your task is to translate the following structured output (in JSON) strictly into the target language specified, preserving the meaning exactly. "
        f"You must respond ONLY in the target language: {language}. Do not use any other language, and do not provide any explanation or commentary.\n\n"
        f"Structured Output (JSON):\n{json.dumps(structured_output, ensure_ascii=False, indent=2)}\n\nTarget Language: {language}\n\n"
        f"For example, if the target language is Hindi, your entire response must be in Hindi. If the target language is Gujarati, your entire response must be in Gujarati.\n"
        f"Double-check that your output is ONLY in the requested language and covers all content.\n"
        f"Return only the translated text, no explanation."
    )
    try:
        response = llm.invoke(prompt)
        return str(response)
    except Exception as e:
        return f"Error generating translation: {str(e)}"

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
        # Accept both dict and JSON string for structured_output
        result = call_llm_translate(payload.structured_output, payload.language)
        return {"translated_output": result, "language": payload.language}
    except Exception as e:
        logging.error(f"Translation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate-feedbacks")
async def translate_feedbacks(payload: FeedbackTranslationPayload):
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return {"translated_feedbacks": "Google API key is required. Please set GOOGLE_API_KEY in your .env file.", "language": payload.language}
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=512,
                timeout=30,
                max_retries=2,
                google_api_key=google_api_key,
            )
        except Exception as e:
            return {"translated_feedbacks": f"Error initializing LLM: {str(e)}", "language": payload.language}

        # Parse feedbacks from JSON string if needed
        feedbacks = payload.feedbacks
        if isinstance(feedbacks, str):
            try:
                feedbacks = json.loads(feedbacks)
            except Exception:
                pass

        prompt = (
            f"You are a professional translator. Your task is to translate the following feedbacks (in JSON) strictly into the target language specified, preserving the meaning exactly. "
            f"You must respond ONLY in the target language: {payload.language}. Do not use any other language, and do not provide any explanation or commentary.\n\n"
            f"Feedbacks (JSON):\n{json.dumps(feedbacks, ensure_ascii=False, indent=2)}\n\nTarget Language: {payload.language}\n\n"
            f"For example, if the target language is Hindi, your entire response must be in Hindi. If the target language is Gujarati, your entire response must be in Gujarati.\n"
            f"Double-check that your output is ONLY in the requested language and covers all content.\n"
            f"Return only the translated text, no explanation."
        )
        try:
            response = llm.invoke(prompt)
            return {"translated_feedbacks": str(response), "language": payload.language}
        except Exception as e:
            return {"translated_feedbacks": f"Error generating translation: {str(e)}", "language": payload.language}
    except Exception as e:
        logging.error(f"Feedback Translation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
