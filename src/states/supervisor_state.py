from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing import Sequence, Annotated, List, Literal, Optional, TypedDict
import operator
from langgraph.graph.message import add_messages


class Fact(BaseModel):
    """
    An atomic unit of knowledge, extracted from raw research notes 
    and stored in our structured knowledge base.
    """
    # The core factual statement itself, extracted from a source.
    content: str = Field(description="The factual statement")

    # We must track the provenance of every fact for traceability and citations in the final report.
    source_url: str = Field(description="Where this fact came from")

    # A confidence score allows the system to weigh more credible sources higher during synthesis.
    confidence_score: int = Field(description="1-100 confidence score based on source credibility")

    # This flag allows our Red Team to mark facts that are contradicted by other evidence, 
    # a key part of our self-correction loop.
    is_disputed: bool = Field(default=False, description="If this fact conflicts with others")


class Critique(BaseModel):
    """
    A structured model for adversarial feedback from the Red Team or
    other quality control agents.
    """
    # Tracks which agent generated the critique (e.g., "Red Team", "Safety Filter") for accountability.
    author: str 

    # The specific logical fallacy, bias, or factual error that was found in the draft.
    concern: str

    # A 1-10 score to quantify the severity of the issue, allowing the Supervisor agent to prioritize its actions.
    severity: int 

    # A flag to track whether a critique has been addressed in a subsequent revision of the draft.
    addressed: bool = Field(default=False, description="Has the supervisor fixed this?")


class QualityMetric(TypedDict):
    """A TypedDict for storing a snapshot of the draft's quality at a specific iteration."""
    # The programmatic quality score calculated by our self-evolution evaluator.
    score: float
    # The textual feedback from the evaluator explaining the score.
    feedback: str
    # The iteration number at which this score was recorded, for tracking progress over time.
    iteration: int


class SupervisorState(TypedDict):
    """
    The advanced, hierarchical state for the main Supervisor agent, 
    the central workbench for the diffusion process.
    """
    # A standard field for accumulating the conversational history with the Supervisor.
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # These are the core artifacts of the research process that the Supervisor manages.
    research_brief: str
    draft_report: str
    
    # This is a key memory management design. 'raw_notes' is a temporary, high-volume buffer
    # for unprocessed search results. 'knowledge_base' is the permanent, structured, and pruned storage.
    raw_notes: Annotated[List[str], operator.add] 
    knowledge_base: Annotated[List[Fact], operator.add]
    
    # A simple counter to prevent infinite loops in our iterative process.
    research_iterations: int
    
    # These fields manage the self-correction and adversarial feedback loops.
    active_critiques: Annotated[List[Critique], operator.add]
    quality_history: Annotated[List[QualityMetric], operator.add]

    # A boolean flag that the Evaluator can set to signal to the Supervisor 
    # that the draft quality is unacceptably low.
    needs_quality_repair: bool
    

class EvaluationResult(BaseModel):
    """A Pydantic schema for the structured output of our programmatic quality evaluator."""

    # A 0-10 score on how well the draft covers all aspects of the research brief.
    comprehensiveness_score: int = Field(description="0-10 score on coverage")

    # A 0-10 score on whether the claims in the draft are factually grounded.
    accuracy_score: int = Field(description="0-10 score on factual grounding")

    # A 0-10 score on the logical flow and readability of the draft.
    coherence_score: int = Field(description="0-10 score on flow")

    # Actionable feedback for the researcher on how to improve the draft.
    specific_critique: str = Field(description="Actionable feedback for the researcher")
    