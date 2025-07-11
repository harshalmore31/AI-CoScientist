"""
AI-CoScientist: A multi-agent framework for collaborative scientific research.

This package implements the "Towards an AI Co-Scientist" methodology with
tournament-based hypothesis evolution, peer review systems, and intelligent
agent orchestration.
"""

__version__ = "1.0.0"
__author__ = "The Swarm Corporation"
__description__ = "A multi-agent AI framework for collaborative scientific research, implementing tournament-based hypothesis evolution and peer review systems"

from .main import (
    AIScientistFramework,
    Hypothesis,
    AgentRole,
    ReviewScores,
    DetailedFeedback,
    HypothesisReview,
    WorkflowResult,
    ExecutionMetrics,
    AgentExecutionMetrics,
    SimilarityCluster,
    ProximityAnalysisResult,
    TournamentJudgment,
)

__all__ = [
    "AIScientistFramework",
    "Hypothesis",
    "AgentRole",
    "ReviewScores",
    "DetailedFeedback",
    "HypothesisReview",
    "WorkflowResult",
    "ExecutionMetrics",
    "AgentExecutionMetrics",
    "SimilarityCluster",
    "ProximityAnalysisResult",
    "TournamentJudgment",
    "__version__",
    "__author__",
    "__description__",
]
