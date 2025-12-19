from langchain_core.messages import SystemMessage, HumanMessage

from src.states.supervisor_state import SupervisorState, Critique
from src.models.hf_models import get_hf_model

# We define a specialized, powerful model for our critic.
# In a real system, you might use a model specifically fine-tuned for critical analysis.
critic_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")
async def red_team_node(state: SupervisorState) -> dict:
    """
    This node represents the 'Red Team' agent. It runs in parallel to other steps,
    critiquing the current draft to find logical flaws and biases.
    """

    # 1. We get the current draft report from the state.
    draft = state.get("draft_report", "")
    
    # 2. We add a guardrail: the Red Team only activates if the draft is substantial enough to critique.
    if not draft or len(draft) < 50:
        return {} 

    # 3. This is the adversarial prompt. It explicitly instructs the model to be "NOT helpful"
    #    and to focus on specific types of errors like missing citations and logical leaps.
    prompt = f"""
    You are the 'Red Team' Adversary. 
    The researcher has written the following draft report. 
    
    <Draft>
    {draft}
    </Draft>
    
    Your goal is NOT to be helpful. Your goal is to find:
    1. Claims that lack citations or are not supported by the evidence.
    2. Logical leaps where the conclusion does not follow from the premises.
    3. Significant bias or a failure to consider alternative viewpoints.
    
    If the draft is solid and has no major logical or factual issues, output exactly "PASS".
    If there are issues, output a specific, harsh, and actionable critique describing the errors.
    """

    # 4. We invoke our powerful critic model.
    response = await critic_model.ainvoke([HumanMessage(content=prompt)])
    content = response.content

    # 5. If the model outputs "PASS", no critique is needed, and this node returns an empty update.
    if "PASS" in content and len(content) < 20:
        return {}

    # 6. If a flaw is found, we create a structured 'Critique' object.
    critique = Critique(
        author="Red Team Adversary",
        concern=content,
        severity=8, # We default to a high severity for Red Team findings.
        addressed=False
    )

    # 7. We return two updates: we add the formal critique to the 'active_critiques' list,
    #    and we also inject a high-priority SystemMessage directly into the Supervisor's message history.
    return {
        "active_critiques": [critique],
        "supervisor_messages": [
            SystemMessage(content=f"⚠️ ADVERSARIAL FEEDBACK DETECTED: {content}")
        ]
    }