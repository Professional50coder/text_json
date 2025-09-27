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

class EvidenceItem(BaseModel):
    claim: str
    supporting_assets: List[str] = []  # e.g., ["audio_01@00:12", "img_02"]
    confidence: Optional[float] = None  # 0..1
    excerpt: Optional[str] = None  # short quoted excerpt (if available)


class SectionScores(BaseModel):
    problem: Optional[int] = Field(None, ge=0, le=5)
    market: Optional[int] = Field(None, ge=0, le=5)
    value_proposition: Optional[int] = Field(None, ge=0, le=5)
    business_model: Optional[int] = Field(None, ge=0, le=5)
    team: Optional[int] = Field(None, ge=0, le=5)
    traction: Optional[int] = Field(None, ge=0, le=5)
    funding_readiness: Optional[int] = Field(None, ge=0, le=5)


class BusinessPlanAnalysis(BaseModel):
    submission_id: str
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_provenance: Dict[str, Any] = {}  # e.g. {"llm_model":"gpt-4o-mini","prompt_version":"v1"}

    # overall confidence 0..1
    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    # structured scores
    scores: SectionScores = Field(default_factory=SectionScores)

    # textual outputs
    strengths: List[str] = []
    weaknesses: List[str] = []
    prioritized_actions: List[str] = []  # prioritized next steps / experiments
    red_flags: List[str] = []
    risk_assessment: Optional[str] = None
    automated_feedback: Optional[str] = None  # ~200-400 words critique summary

    # evidence map ties claims to assets/excerpts
    evidence_map: List[EvidenceItem] = []

    # optional numerical KPIs extracted (if any)
    extracted_kpis: Dict[str, Any] = {} 