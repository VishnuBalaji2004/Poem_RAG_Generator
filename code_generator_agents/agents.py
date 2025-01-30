#maximum changes done

from langchain.agents import create_react_agent, AgentExecutor
import tools
import models
import prompts


#### AGENT ####
def generate_code_with_agent(language, problem_statement):
    """
    Generate a poem using a LangChain agent with a Shakespearean-style prompt.

    Args:
        topic(str) : Topic for the poem

    Returns:
        str: Generated poem.
    """
    # Define tools for the agent
    tools_list = [tools.code_generator_tool()]

    # Initialize the agent with the Shakespearean-style prompt template
    prompt_template = prompts.code_generator_agent()
    llm = models.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, handle_parsing_errors=True, verbose=True, stop_sequence=True, max_iterations=3)

    # Agent interaction
    response = agent_executor.invoke({
    "input": f"Language: {language}, Problem: {problem_statement}"
})


    # response = agent_executor.invoke({
    #     "programming_language": language,
    #     "problem_statement": problem_statement
    #     })
    return response

#### AGENT WITH RAG ####
def generate_code_with_rag_agent(language, problem_statement, vector):
    """
    Generate code using a LangChain agent with Retrieval-Augmented Generation (RAG).

    Args:
        language (str): Programming language for the code
        problem_statement (str): Problem statement for the code
        vector (object): Instance of vector store

    Returns:
        str: Generated code
    """
    # Define tools for the agent
    tools_list = [
        tools.rag_retriever_tool(vector),
        tools.code_generator_tool()  # Ensure to call the function to get the tool object
    ]

    # Initialize the agent with the RAG-enabled prompt template
    prompt_template = prompts.code_generator_agent_with_rag()
    llm = models.create_chat_groq_model()
    agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, handle_parsing_errors=True, verbose=True, stop_sequence=True, max_iterations=3)

    # Agent interaction
    response = agent_executor.invoke({
        "input":"write a {language} program to {problem_statement}"
    })
    return response


# def generate_code_with_rag_agent(language, problem_statement, vector):
#     """
#     Generate a poem using a LangChain agent with Retrieval-Augmented Generation (RAG).

#     Args:
#         topic (str): Topic for the poem
#         vector (object): Instance of vector store

#     Returns:
#         str: Generated poem
#     """
#     # Define tools for the agent
#     tools_list = [
#         tools.rag_retriever_tool(vector),
#         tools.code_generator_tool(lambda problem_statement: problem_statement)
#     ]

#     # Initialize the agent with the RAG-enabled prompt template
#     prompt_template = prompts.code_generator_agent_with_rag()
#     llm = models.create_chat_groq_model()
#     agent = create_react_agent(tools=tools_list, llm=llm, prompt=prompt_template)
#     agent_executor = AgentExecutor(agent=agent, tools=tools_list, handle_parsing_errors=True, verbose=True, stop_sequence=True, max_iterations=3)

#     # Agent interaction
#     response = agent_executor.invoke({
#         "input":"write a {language} program to {problem_statement}"
#         })
#     return response