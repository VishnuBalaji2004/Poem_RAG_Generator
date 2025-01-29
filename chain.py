from model import create_chat_gorq
from prompts import poem_generator_prompt

def generate_poem(topic):
    """
    Function to initialize chat gorq
    Args:
        topic(str) - topic of the poem
    Returns:
        response.content (str)
    """
    prompt_template = poem_generator_prompt()
    llm =create_chat_gorq()
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "topic" : topic
    })
    return response.content
