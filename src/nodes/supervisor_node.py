from typing import Literal
from langgraph.types import Command
from langgraph.graph import END
import asyncio
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.states.supervisor_state import SupervisorState, QualityMetric, EvaluationResult
from src.models.hf_models import get_hf_model
from src.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt
from src.helpers import get_today_str, get_notes_from_tool_calls
from src.tools.supervisor_tools import think_tool, refine_draft_report, ConductResearch, ResearchComplete
from src.graphs.researcher_graph import researcher_agent



max_concurrent_researchers = 3
max_researcher_iterations = 5


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    The 'Brain' of the diffusion process. This node analyzes the current state,
    including any critical feedback, and decides on the next set of actions (tool calls).
    """
    # 1. We get the current message history for the supervisor.
    supervisor_messages = state.get("supervisor_messages", [])
    
    # 2. We format the main system prompt with the diffusion algorithm instructions.
    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 3. DYNAMIC CONTEXT INJECTION: We check for and inject any unaddressed adversarial feedback.
    # This is a critical self-correction mechanism.
    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in unaddressed])
        intervention = SystemMessage(content=f"""
        CRITICAL INTERVENTION REQUIRED.
        The following issues were detected by the Adversarial Team in your draft:
        {critique_text}
        
        You MUST address these issues in your next step.
        If the critique says citations are missing, call 'ConductResearch' to find them.
        If the critique says logic is flawed, call 'think_tool' to plan a fix.
        """)
        messages.append(intervention)

    # 4. We also inject a warning if the programmatic quality score was low in the last iteration.
    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(content="PREVIOUS DRAFT QUALITY WAS LOW (Score < 7/10). Focus on finding new sources and citing them."))

    # 5. We invoke the tool-bound supervisor model to get its next plan.
    tools = [think_tool, refine_draft_report, ConductResearch, ResearchComplete]  # Add relevant tools here
    supervisor_model_with_tools = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct").bind_tools(tools=tools) # Add relevant tools here
    response = await supervisor_model_with_tools.ainvoke(messages)

    # 6. We return a Command to proceed to the 'supervisor_tools' node to execute the plan.
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            "needs_quality_repair": False # We reset the repair flag after warning the model.
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["red_team", "context_pruner", "__end__"]]:
    """
    The 'Hands' of the Supervisor. This node executes the planned tool calls, including
    fanning out to parallel research sub-graphs and running the denoising step.
    """

    # 1. We get the most recent message, which contains the tool calls to be executed.
    most_recent_message = state.get("supervisor_messages", [])[-1]
    
    # 2. We check for exit conditions for the entire diffusion loop.
    exceeded_iterations = state.get("research_iterations", 0) >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls)


    if exceeded_iterations or no_tool_calls or research_complete:

        # If exiting, we prepare the final, curated notes for the report writer.
        # We prioritize the structured Knowledge Base, but fall back to raw notes if it's empty.
        kb_notes = [f"{f.content} (Confidence: {f.confidence_score})" for f in state.get("knowledge_base", [])]
        if not kb_notes: kb_notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))

        # We return a Command to END this sub-graph and pass the final notes up to the main graph.
        return Command(goto=END, update={"notes": kb_notes, "research_brief": state.get("research_brief", "")})

    # 3. We separate the different types of tool calls for specialized handling.
    conduct_research_calls = [t for t in most_recent_message.tool_calls if t["name"] == "ConductResearch"]
    refine_report_calls = [t for t in most_recent_message.tool_calls if t["name"] == "refine_draft_report"]
    think_calls = [t for t in most_recent_message.tool_calls if t["name"] == "think_tool"]
    
    tool_messages = []
    all_raw_notes = []
    draft_report = state.get("draft_report", "")
    updates = {}

    # 4. Handle 'think_tool' calls synchronously.
    for tool_call in think_calls:
        observation = think_tool.invoke(tool_call["args"])
        tool_messages.append(ToolMessage(content=observation, name="think_tool", tool_call_id=tool_call["id"]))

    # 5. Handle 'ConductResearch' calls by fanning out to our research sub-graph in parallel.
    if conduct_research_calls:

        # We create a list of coroutines, one for each research task.
        coros = [researcher_agent.ainvoke({"researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])], "research_topic": tc["args"]["research_topic"]}) for tc in conduct_research_calls]

        # 'asyncio.gather' runs all the research sub-graphs concurrently.
        results = await asyncio.gather(*coros)
        for result, tool_call in zip(results, conduct_research_calls):

            # We append the clean, compressed research as a ToolMessage for the Supervisor's context.
            tool_messages.append(ToolMessage(content=result.get("compressed_research", ""), name=tool_call["name"], tool_call_id=tool_call["id"]))

            # We also collect the raw, uncompressed notes to be processed by our context pruner.
            all_raw_notes.extend(result.get("raw_notes", []))

    # 6. Handle 'refine_draft_report' calls. This is the core denoising and self-evaluation step.
    for tool_call in refine_report_calls:
        kb = state.get("knowledge_base", [])
        kb_str = "CONFIRMED FACTS:\n" + "\n".join([f"- {f.content}" for f in kb]) if kb else "\n".join(get_notes_from_tool_calls(state.get("supervisor_messages", [])))
        new_draft = refine_draft_report.invoke({"research_brief": state.get("research_brief", ""), "findings": kb_str, "draft_report": state.get("draft_report", "")})
        
        # --- CRITICAL STEP: The Self-Evolution Evaluation ---
        eval_result = evaluate_draft_quality(research_brief=state.get("research_brief", ""), draft_report=new_draft)
        avg_score = (eval_result.comprehensiveness_score + eval_result.accuracy_score) / 2
        
        # We include the quality score directly in the tool message, so the Supervisor sees it.
        tool_messages.append(ToolMessage(content=f"Draft Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.specific_critique}", name=tool_call["name"], tool_call_id=tool_call["id"]))
        draft_report = new_draft
        
        # We log the metric to our history and set the repair flag if the score is low.
        updates["quality_history"] = [QualityMetric(score=avg_score, feedback=eval_result.specific_critique, iteration=state.get("research_iterations", 0))]
        if avg_score < 7.0: updates["needs_quality_repair"] = True

    # 7. Prepare the final state updates for this iteration.
    updates["supervisor_messages"] = tool_messages
    updates["raw_notes"] = all_raw_notes
    updates["draft_report"] = draft_report
    
    # 8. FAN OUT to the self-correction nodes (Red Team and Context Pruner) in parallel.
    return Command(goto=["red_team", "context_pruner"], update=updates)


def evaluate_draft_quality(research_brief: str, draft_report: str) -> EvaluationResult:
    """
    This function implements the 'Self-Evolution' scoring mechanism. It acts as an
    LLM-as-a-judge, programmatically evaluating the quality of a draft against the original brief.
    """

    # We create a prompt that asks the judge model to be an extremely critical Senior Research Editor.
    eval_prompt = f"""
    You are a Senior Research Editor. Your standards are exceptionally high. Evaluate this draft report against the research brief.
    
    <Research Brief>
    {research_brief}
    </Research Brief>
    
    <Draft Report>
    {draft_report}
    </Draft Report>
    
    Be extremely critical. High scores (8+) should be reserved for truly excellent, comprehensive, and well-cited work. 
    Focus your evaluation on these key areas:
    1. **Comprehensiveness:** Does the draft fully address all parts of the research brief? Are there significant gaps?
    2. **Accuracy & Grounding:** Are the claims specific and well-supported? Look for vague statements that need citations.
    3. **Coherence & Structure:** Is the report well-organized and easy to follow? Is the language clear and professional?
    
    Provide specific, actionable critique for the researcher.
    """
    
    # We bind our EvaluationResult schema to our judge model.
    # Our judge model
    judge_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")
    structured_judge = judge_model.with_structured_output(EvaluationResult)

    # We invoke the judge to get the structured quality score.
    return structured_judge.invoke([HumanMessage(content=eval_prompt)])