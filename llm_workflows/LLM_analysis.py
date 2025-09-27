from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import BusinessPlanAnalysis
import json
import os
from typing import Dict, Any

def analyze_business_plan(json_data: Any) -> BusinessPlanAnalysis:
    """
    Analyze business plan JSON data using LLM and return structured BusinessPlanAnalysis output.
    
    Args:
        json_data: Dictionary containing business plan information to analyze
        
    Returns:
        BusinessPlanAnalysis: Structured analysis with scores, feedback, and recommendations
    """
    # Get API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("Google API key is required. Please set GOOGLE_API_KEY in your .env file.")
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=google_api_key,
    )
    
    # Get structured output using the BusinessPlanAnalysis class
    structured_llm = llm.with_structured_output(BusinessPlanAnalysis)
    
    # Create analysis prompt from the JSON data
    prompt = f"""
    You are an expert business plan analyst. Analyze the following business plan data and provide a comprehensive analysis according to the BusinessPlanAnalysis schema.
    
    Evaluate the business plan across these dimensions:
    - Problem identification and market opportunity (0-5 scale)
    - Market understanding and sizing (0-5 scale) 
    - Value proposition clarity (0-5 scale)
    - Business model viability (0-5 scale)
    - Team strength and capabilities (0-5 scale)
    - Traction and validation evidence (0-5 scale)
    - Funding readiness (0-5 scale)
    
    Provide:
    - Specific strengths and weaknesses
    - Prioritized action items for improvement
    - Red flags or major concerns
    - Overall risk assessment
    - Overall confidence in the plan (0.0-1.0)
    
    Business Plan Data:
    {json.dumps(json_data, indent=2)}
    
    Please provide a thorough analysis structured according to the BusinessPlanAnalysis schema.
    """
    
    # Get structured analysis response
    response = structured_llm.invoke(prompt)
    # Minimal post-processing for dict fields
    data = response if isinstance(response, dict) else response.dict() if hasattr(response, 'dict') else response
    # Ensure model_provenance is a dict
    if 'model_provenance' in data and not isinstance(data['model_provenance'], dict):
        data['model_provenance'] = {'value': data['model_provenance']}
    # Ensure extracted_kpis is a dict
    if 'extracted_kpis' in data and not isinstance(data['extracted_kpis'], dict):
        data['extracted_kpis'] = {'value': data['extracted_kpis']}
    return BusinessPlanAnalysis(**data)
