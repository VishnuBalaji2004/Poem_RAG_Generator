# maximum changes done

from langchain_core.output_parsers import StrOutputParser
import models
import prompts
import vectordb


#### GENERATION ####
def generate_code_chain(language, problem_statement):
    """
    Generate code using basic prompt LLM chain

    Args:
        language- language of the code 
        problem_statement - problem statement to be solved

    Returns:
        response.content -> str
    """
        
    llm = models.create_chat_groq_model()

    prompt_template = prompts.code_generator_prompt()
    # prompt_template = prompts.code_generator_prompt_from_hub()

    chain = prompt_template | llm

    response = chain.invoke({
        "input": f"Language: {language}, Problem Statement: {problem_statement}"
    })
    return response.content


#### RETRIEVAL and GENERATION ####
def generate_code_rag_chain(language, problem_statement, vector):
    """
    Creates a RAG chain for retrieval and generation.

    Args:
        language (str): Language of the code.
        problem_statement (str): Problem statement to be solved.
        vectorstore (object): Instance of vector store.

    Returns:
        str: Generated code.
    """
    # Prompt
    prompt = prompts.code_generator_rag_prompt()

    # LLM
    llm = models.create_chat_groq_model()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the combined query string
    query = f"Language: {language}, Problem Statement: {problem_statement}"

    # Retrieve documents based on the query
    retriever = vectordb.retrieve_from_chroma(query, vectorstore=vector)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({
        "context": format_docs(retriever),
        "input": query,
    })    

    return response

#
# def generate_code_rag_chain(language, problem_statement, vector):
#     """
#     Creates a RAG chain for retrieval and generation.

#     Args:
        
#         language - language of the code
#         problem_statement - problem statement to be solved
#         vectorstore ->  Instance of vector store 

#     Returns:
#         rag_chain -> rag chain
#     """
#     # Prompt
#     prompt = prompts.code_generator_rag_prompt()

#     # LLM
#     llm = models.create_chat_groq_model()

#     # Post-processing
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)
    
#     retriever = vectordb.retrieve_from_chroma(language + problem_statement, vectorstore=vector)
#     # Chain
#     rag_chain = prompt| llm | StrOutputParser()

#     response = rag_chain.invoke({
#         "context" : format_docs(retriever),
#         "programming_language": language,
#         "problem_statement": problem_statement
#     })    

#     return response
