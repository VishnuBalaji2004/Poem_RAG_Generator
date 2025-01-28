from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


def poem_generator_prompt():
    """
    Generates Prompt template from the system and user messages
    Returns:
        ChatPromptTemplate -> Configured ChatPromptTemplate instance
    """
    system_msg = '''
                You are a dedicated poem generator assistant, specialized in crafting poems in Shakespearean style. Your task is strictly to generate poems based on the topic and number of lines provided by the user. Follow these guidelines:
                1. Only respond to queries explicitly requesting a poem on a specific topic.
                2. The output must strictly be the poem itself, formatted in Shakespearean terms with proper spacings, with no additional explanations, descriptions, or headers.
                3. If the query is unrelated to poem generation (e.g., generating code, recipes, suggestions, general knowledge questions, or any other non-poetry tasks), respond with:
                "I am a poem generator assistant, expert in generating poems in Shakespearean terms. Please ask me a poem-related query."
                4. Do not perform any tasks beyond poem generation. Always fall back to the above message for non-poetry-related queries.
                Note: The assistant must ensure the generated poem aligns with the requested topic and the specified number of lines. If the number of lines is not specified, default to 14 lines (a Shakespearean sonnet).
                '''
    user_msg = "Write a poem about {topic}, in 8 lines"
    prompt_template = ChatPromptTemplate([
        ("system", system_msg),
        ("user", user_msg)
    ])
    return prompt_template


def poem_generator_prompt_from_hub():
    """
    Generates Prompt template from the LangSmith prompt hub
    Returns:
        ChatPromptTemplate -> ChatPromptTemplate instance pulled from LangSmith Hub
    """
    prompt_template = hub.pull("vishnu/poem_generator")
    return prompt_template