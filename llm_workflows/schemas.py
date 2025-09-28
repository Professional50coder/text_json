from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
import uuid

class BusinessPlanDetails(BaseModel):
    title: Optional[str] = None
    tagline: Optional[str] = None
    vision: Optional[str] = None
    mission: Optional[str] = None
    language: Optional[str] = None
    stage: Optional[str] = None  # Idea / Prototype / Pre-revenue / Revenue
    summary: Optional[str] = None

    summary: Optional[str] = None  # Combine one_line_summary, executive_summary, explain_like_im_12 into one text field

    # Problem and solution combined sections
    problem_and_customer: Optional[str] = None  # combines problem_statement, target_customer, evidence_of_problem
    solution_and_features: Optional[str] = None  # combines solution_description, key_features, usp

    # Market and competition combined
    market_and_competitors: Optional[str] = None  # combines market_segments, competitors
    channels_and_revenue: Optional[str] = None  # combines channels, revenue_streams, price_points, cost_structure

    # Operations and team combined
    operations_and_team: Optional[str] = None  # combines operations_summary, suppliers_and_logistics, location_of_operations, team_summary, missing_skills

    # Traction and funding combined
    traction_and_funding: Optional[str] = None  # combines traction, ask, funding_required, funding_use

    # Risks and mitigation combined
    risks_and_mitigation: Optional[str] = None  # combines risks, mitigation

    # Impact combined
    social_and_environmental_impact: Optional[str] = None  # combines social_impact, environmental_impact



class BusinessPlanAnalysis(BaseModel):
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Section scores combined into fewer category scores (0 to 5)
    problem_and_market_score: Optional[int] = Field(None, ge=0, le=5)
    value_and_model_score: Optional[int] = Field(None, ge=0, le=5)
    team_and_traction_score: Optional[int] = Field(None, ge=0, le=5)
    funding_readiness_score: Optional[int] = Field(None, ge=0, le=5)
    market_feasibility_score: Optional[int] = Field(None, ge=0, le=5)
    financial_feasibility_score: Optional[int] = Field(None, ge=0, le=5)
    technical_feasibility_score: Optional[int] = Field(None, ge=0, le=5)

    # Textual feedback fields
    strengths: List[str] = []
    weaknesses: List[str] = []
    prioritized_actions: List[str] = []
    red_flags: List[str] = []
    risk_assessment: Optional[str] = None
    automated_feedback: Optional[str] = None

    # Optional textual feedback for feasibility aspects
    market_feasibility_feedback: Optional[str] = None
    financial_feasibility_feedback: Optional[str] = None
    technical_feasibility_feedback: Optional[str] = None

    # Basis or rationale for scoring all scores
    problem_and_market_basis: Optional[str] = None
    value_and_model_basis: Optional[str] = None
    team_and_traction_basis: Optional[str] = None
    funding_readiness_basis: Optional[str] = None
    market_feasibility_basis: Optional[str] = None
    financial_feasibility_basis: Optional[str] = None
    technical_feasibility_basis: Optional[str] = None  # 200-400 word critique summary

    # Extracted numerical KPIs (key performance indicators)
    extracted_kpis: list[str] = []
    news_summary: Optional[str] = None  # Summary of relevant news articles