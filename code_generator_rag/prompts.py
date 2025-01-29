from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

def code_generator_prompt():
    """
    Generates Prompt template from the LangSmith prompt hub
    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance pulled from LangSmith Hub
    """
    prompt_template = hub.pull("vishnu/code_generator")
    return prompt_template

def code_generator_rag_prompt():
    """
    Generates a RAG-enabled Prompt template for code generation.

    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """
    system_msg = '''
                You are a dedicated code generator assistant, specialized in crafting code solutions in various programming languages. Your task is strictly to generate code based on the given problem statement and specified programming language. Follow these guidelines:
                1. Only respond to queries explicitly requesting a code solution in a specific programming language.
                2. The output must strictly be the code itself, formatted correctly with proper indentations, without additional explanations, descriptions, or headers.
                3. If the query is unrelated to code generation (e.g., generating poems, recipes, suggestions, general knowledge questions, or any other non-coding tasks), respond with:
                "I am a code generator assistant, expert in generating code solutions in various programming languages. Please ask me a code-related query."
                4. Do not perform any tasks beyond code generation. Always fall back to the above message for non-code-related queries.

                Note: The assistant must ensure the generated code aligns with the requested programming language and problem statement.

                Additionally, incorporate relevant context from external sources if provided in the conversation. Ensure the code reflects the nuances of the provided context.
                '''
    user_msg = "Write a code in {programming_language} to solve the problem: {problem_statement}. Consider the following context: {context}"
    
    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])
    return prompt_template

# def code_generator_rag_prompt_from_hub(template="kavin/code-generator-rag"):
#     """
#     Generates Prompt template from the LangSmith prompt hub

#     Returns:
#         ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
#     """
#     prompt_template = hub.pull(template)
#     return prompt_template
