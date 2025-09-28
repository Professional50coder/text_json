from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import BusinessPlanDetails
import json
import os
from typing import Dict, Any

def get_structured_business_plan_student(json_data: Dict[str, Any]) -> BusinessPlanDetails:
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
    
    # Get structured output using the BusinessPlanDetails class
    structured_llm = llm.with_structured_output(BusinessPlanDetails)
    
    # Create prompt from the JSON data
    prompt = f"""
    Analyze the following business plan data and extract/structure the information according to the BusinessPlanDetails schema.
    Fill in as many fields as possible based on the provided data. If information is not available, leave fields as None or empty lists as appropriate.
    When giving feedback or structuring the data, always write in second person (use "you" instead of "they").
    The summary field should be in points or bullet format for clarity highlighting key aspects of the business plan.
    the scores should be on a 0-10 scale, where 0 is poor and 10 is excellent on various dimensions of the business plan.
    
    Business Plan Data:
    {json.dumps(json_data, indent=2)}
    
    Please structure this data according to the BusinessPlanDetails schema.
    """
    
    # Get structured response
    response = structured_llm.invoke(prompt)
    
    return response


def get_structured_business_plan_mentor(json_data: Dict[str, Any]) -> BusinessPlanDetails:
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
    
    # Get structured output using the BusinessPlanDetails class
    structured_llm = llm.with_structured_output(BusinessPlanDetails)
    
    # Create prompt from the JSON data
    prompt = f"""
    Analyze the following business plan data and extract/structure the information according to the BusinessPlanDetails schema.
    Fill in as many fields as possible based on the provided data. If information is not available, leave fields as None or empty lists as appropriate.
    The summary field should be in points or bullet format for clarity highlighting key aspects of the business plan.
    The scores should be on a 0-10 scale, where 0 is poor and 10 is excellent on various dimensions of the business plan.

    
    Business Plan Data:
    {json.dumps(json_data, indent=2)}
    
    Please structure this data according to the BusinessPlanDetails schema.
    """
    
    # Get structured response
    response = structured_llm.invoke(prompt)
    
    return response
