from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def create_chat_gorq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
):
    """
    Function to initialize chat gorq
    Returns:
        ChatGroq
    """
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        cache=False
    )



def create_hugging_face_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Creates and returns a configured instance of the HuggingFace embeddings model.

    Args:
        model_name -> str: The model to use (default: "sentence-transformers/all-MiniLM-L6-v").

    Returns:
        HuggingFaceEmbeddings: Configured HuggingFaceEmbeddings model instance
    """
    return HuggingFaceEmbeddings(model_name=model_name)