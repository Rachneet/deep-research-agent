from typing import List
from pydantic import BaseModel

from src.models.hf_models import get_hf_model
from src.states.supervisor_state import Fact, SupervisorState
from langchain_core.messages import HumanMessage, SystemMessage


class FactExtraction(BaseModel):
    """A Pydantic schema for the output of the context pruning/fact extraction agent."""
    # The output is a list of the 'Fact' objects we defined in Part 1.
    new_facts: List[Fact]


async def context_pruning_node(state: SupervisorState) -> dict:
    """
    This node implements 'Context Engineering'. It takes the temporary buffer of raw notes,
    extracts structured facts, adds them to the permanent knowledge base, and then clears the buffer.
    """
    # 1. We get the current raw notes and the permanent knowledge base from the state.
    raw_notes = state.get("raw_notes", [])
    
    # 2. We add a guardrail: this node only runs if there are new raw notes to process.
    if not raw_notes:
        return {}

    # 3. We combine all the new raw notes into a single block of text for processing.
    text_block = "\n".join(raw_notes)

    # 4. We create a prompt that instructs the LLM to act as a Knowledge Graph Engineer.
    prompt = f"""
    You are a Knowledge Graph Engineer.
    
    New Raw Notes from a research agent:
    {text_block[:20000]} 
    
    Your task is to:
    1. Extract all atomic, verifiable facts from the New Raw Notes.
    2. For each fact, identify its source URL.
    3. Assign a confidence score (1-100) based on the credibility of the source.
    4. Ignore any information that is the agent's internal "thinking" or planning.
    
    Return ONLY a valid JSON object with a single key 'new_facts' containing a list of these structured facts.
    """
    
    try:

        # 5. We invoke our structured output LLM to perform the extraction.
        # We'll use a fast, cheaper model for this routine extraction task.
        compressor_model = get_hf_model(model_name="Qwen/Qwen3-4B-Instruct-2507")
        structured_llm = compressor_model.with_structured_output(FactExtraction)
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        new_facts = result.new_facts
        
        # In a more advanced system, we would perform deduplication against the existing knowledge_base here.
        message = f"[SYSTEM] Context Pruned. {len(new_facts)} new facts added to Knowledge Base. Raw notes buffer cleared."
    except Exception as e:

        # If the extraction fails, we create a system message to log the error.
        new_facts = []
        message = f"[SYSTEM] Context Pruning failed: {str(e)}"
    
    # 6. This is the most critical step. We return an update that CLEARS the 'raw_notes' buffer,
    #    appends the 'new_facts' to the permanent 'knowledge_base', and replaces the raw text in the
    #    Supervisor's history with a single, concise system message.
    return {
        "raw_notes": [],   # Clear the raw notes buffer : important!
        "knowledge_base": new_facts,
        "supervisor_messages": [
            SystemMessage(content=message)
        ]
    }