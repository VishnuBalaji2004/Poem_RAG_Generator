# Maximum changes done


from langchain.agents import Tool
import chains
from vectordb import retrieve_from_chroma

def code_generator_tool():
    """
    Generate a tool that can create code

    Args:
        language - language of the code
        problem_statement- problem statement of the code
    Returns:
        Code generator Tool 
    """
    return Tool(
            name="Code Generator",
            func=lambda language, problem_statement: chains.generate_code_chain(language, problem_statement),
            description="Generates a code based on a given language and problem statement.",
        )

def rag_retriever_tool(vector):
    """
    Create a Tool for retrieving relevant documents using RAG

    Args:
        vector (object): The vector store instance.

    Returns:
        Tool: A LangChain Tool object for RAG retrieval.
    """
    return Tool(
            name="RAG Retriever",
            func=lambda topic: "\n\n".join(
                doc.page_content for doc in retrieve_from_chroma(topic, vectorstore=vector)
            ),
            description="Retrieves relevant documents for a given topic using a vector store."
        )
