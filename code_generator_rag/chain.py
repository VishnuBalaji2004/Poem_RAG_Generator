from model import create_chat_gorq
from prompts import code_generator_prompt
import vectordb

def generate_code(language, problem_statement):
    """
    Function to initialize chat gorq
    Args:
        language (str) - programming language for which code is to be generated
        problem_statement (str) - the problem statement for the code generation
    Returns:
        response.content (str) - generated code as a response
    """
    prompt_template = code_generator_prompt()
    llm = create_chat_gorq()
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "programming_language": language,
        "problem_statement": problem_statement
    })
    return response.content

def generate_code_with_rag(language, problem_statement, vectordatabase):
    """
    Function to initialize chat gorq with RAG
    Args:
        language (str) - programming language for which code is to be generated
        problem_statement (str) - the problem statement for the code generation
        vectordatabase (object) - vector database instance for RAG
    Returns:
        response.content (str) - generated code as a response
    """
    prompt_template = code_generator_prompt()
    llm = create_chat_gorq()

    # Fetch relevant information from vector database using the retrieval function
    relevant_data = vectordb.retrieve_from_chroma(
        query={
            "programming_language": language,
            "problem_statement": problem_statement
        }, 
        vectorstore=vectordatabase
    )

    # Incorporate the relevant data into the prompt
    chain = prompt_template | llm
    response = chain.invoke({
        "programming_language": language,
        "problem_statement": problem_statement,
        "relevant_data": relevant_data
    })

    return response.content
