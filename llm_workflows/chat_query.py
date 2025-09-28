from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, Any

router = APIRouter()

class ChatRequest(BaseModel):
	query: str
	transcription: str

class ChatResponse(BaseModel):
	response: str


import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm_response(query: str, transcription: str) -> str:
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

	# Prepare prompt from query and transcript
	# transcription is now always a direct string (business plan text)
	transcript_text = transcription if isinstance(transcription, str) else transcription.get('json_data', '')
	prompt = f"You are an expert assistant. Based on the following business plan transcript, answer the user's query.\n\nTranscript:\n{transcript_text}\n\nQuery: {query}\n\nProvide a concise and relevant answer."

	try:
		response = llm.invoke(prompt)
		return str(response)
	except Exception as e:
		return f"Error generating response: {str(e)}"

import json
from fastapi import HTTPException

@router.post("/chat", response_model=ChatResponse)
async def chat_api(payload: ChatRequest):
	response = get_llm_response(payload.query, payload.transcription)
	return ChatResponse(response=response)
