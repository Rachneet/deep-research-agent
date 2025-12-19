"""
The final_report_generation node takes the notes (the structured, denoised facts from our knowledge_base) 
and the final draft_report from the Supervisor loop and performs one last, high-quality synthesis.

This separation of concerns is important because the Supervisor loop is optimized for iterative research 
and refinement, while this final node is optimized for high-quality, long-form generation.
"""

from src.helpers import get_today_str
from src.prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from src.models.hf_models import get_hf_model
from src.states.agent_state import AgentState
from langchain_core.messages import HumanMessage


# We use our most powerful writer model for this final, high-stakes generation task.
writer_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct", max_tokens=40000) # Using a large max_tokens for comprehensive reports.

async def final_report_generation(state: AgentState):
    """
    The final node in our master graph. It takes all the curated artifacts from the
    Supervisor loop and generates the final, polished report.
    """
    # 1. We retrieve the final, curated notes from the state.
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    # 2. We format our master prompt with all the necessary context.
    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("messages", [HumanMessage(content="")])[-1].content # Pass the original user request for context
    )

    # 3. We invoke our powerful writer model to generate the final report.
    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    # 4. We update the state with the final_report and a user-facing message.
    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }
