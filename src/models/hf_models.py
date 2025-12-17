from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# Initialize the HuggingFace model endpoint
def get_hf_model(model_name: str = "Qwen/Qwen3-4B-Instruct-2507", **kwargs) -> ChatHuggingFace:
    """A wrapper around the HuggingFace LLM endpoint for consistent usage across agents."""

    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            model=model_name,
        ),
        **kwargs
    )
    return llm