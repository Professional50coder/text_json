from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import os

class ImprovementSuggestion(BaseModel):
    section: str  # Which part of the business plan
    priority: str  # High, Medium, Low
    current_issue: str  # What's currently missing or weak
    specific_action: str  # Exact steps to improve
    why_important: str  # Impact of this improvement
    resources_needed: Optional[str] = None  # What they need to complete this

class BusinessPlanFeedback(BaseModel):
    submission_id: str
    feedback_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Overall assessment
    current_strength_level: str  # Beginner, Intermediate, Advanced
    overall_completeness: int = Field(..., ge=0, le=100)  # Percentage complete
    
    # Prioritized improvements
    high_priority_improvements: List[ImprovementSuggestion] = []
    medium_priority_improvements: List[ImprovementSuggestion] = []
    low_priority_improvements: List[ImprovementSuggestion] = []
    
    # Student-focused guidance
    next_steps_this_week: List[str] = []  # Immediate actions
    research_assignments: List[str] = []  # Things to investigate
    questions_to_answer: List[str] = []  # Self-reflection prompts
    
    # Encouragement and progress
    what_youre_doing_well: List[str] = []
    motivational_note: str
    estimated_hours_to_improve: Optional[int] = None

def generate_student_feedback(json_data: Any) -> BusinessPlanFeedback:
    """
    Generate constructive feedback for students to improve their business plan.
    
    Args:
        json_data: Dictionary or string containing student's business plan information
        
    Returns:
        BusinessPlanFeedback: Structured feedback with actionable improvement suggestions
    """
    # Get API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("Google API key is required. Please set GOOGLE_API_KEY in your .env file.")

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,  # Slightly higher for more creative feedback
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=google_api_key,
    )
    
    # Get structured output using the BusinessPlanFeedback class
    structured_llm = llm.with_structured_output(BusinessPlanFeedback)
    
    # Create student-focused feedback prompt
    prompt = f"""
    You are a supportive business plan mentor helping a student improve their business plan.
    Your goal is to provide constructive, actionable feedback that helps them learn and grow.
    When giving feedback, always write in second person (use "you" instead of "they").
    Analyze the student's business plan and provide feedback that:
    ASSESSES your current level (Beginner/Intermediate/Advanced)
    IDENTIFIES what you are doing well (to build your confidence)
    PRIORITIZES improvements (High/Medium/Low priority)
    PROVIDES specific, actionable steps you can take
    SUGGESTS research assignments and self-reflection questions
    ESTIMATES how much work is needed
    ENCOURAGES you with a motivational note
    Focus on:
    Clear, specific actions you can take this week
    Questions you should ask yourself or potential customers
    Resources you might need (interviews, research, data)
    Why each improvement matters for your success
    Building your entrepreneurial thinking skills
    Be constructive, encouraging, and educational. Help you understand not just WHAT to improve, but also HOW and WHY.
    
    Student's Business Plan:
    {json.dumps(json_data, indent=2)}
    
    Please provide structured feedback to help this student improve their business plan.
    """
    
    # Get structured feedback response
    response = structured_llm.invoke(prompt)
    
    return response

# def generate_quick_tips(json_data: Dict[str, Any]) -> str:
#     """
#     Generate a quick text summary of the top 3 improvement tips.
    
#     Args:
#         json_data: Dictionary containing student's business plan information
        
#     Returns:
#         str: Concise improvement tips as a text string
#     """
#     # Initialize the LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.2,
#         max_tokens=200,
#         timeout=None,
#         max_retries=2,
#     )
    
#     prompt = f"""
#     Based on this business plan data, provide exactly 3 quick improvement tips.
#     Keep each tip to 1-2 sentences. Focus on the most impactful changes.
    
#     Format as:
#     1. [Tip about most critical improvement]
#     2. [Tip about second priority]
#     3. [Tip about third priority]
    
#     Business Plan: {json.dumps(json_data, indent=2)}
#     """
    
#     response = llm.invoke(prompt)
#     return response.content
