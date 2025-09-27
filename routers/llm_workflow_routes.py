from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from llm_workflows.structured_template import get_structured_business_plan
from llm_workflows.LLM_analysis import analyze_business_plan
from llm_workflows.plan_feedback import generate_student_feedback
from llm_workflows.schemas import BusinessPlanDetails, BusinessPlanAnalysis
from llm_workflows.plan_feedback import BusinessPlanFeedback
import json

class LLMRequest(BaseModel):
    json_data: Any

router = APIRouter(prefix="/llm-workflow", tags=["LLM Workflows"])


@router.post("/plan-feedback", response_model=BusinessPlanFeedback)
async def generate_plan_feedback(request: LLMRequest):
    """
    Generate student feedback and improvement suggestions for business plan.
    
    Takes business plan JSON data and returns structured feedback with actionable improvements.
    Requires Google API key in request or GOOGLE_API_KEY environment variable.
    """
    try:
        result = generate_student_feedback(request.json_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")

@router.post("/llm-analysis", response_model=BusinessPlanAnalysis)
async def analyze_business_plan_endpoint(request: LLMRequest):
    """
    Analyze business plan and provide comprehensive evaluation.
    
    Takes business plan JSON data and returns structured analysis with scores, strengths, and recommendations.
    Requires Google API key in request or GOOGLE_API_KEY environment variable.
    """
    try:
        result = analyze_business_plan(request.json_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing business plan: {str(e)}")
