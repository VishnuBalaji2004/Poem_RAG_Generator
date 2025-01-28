from dotenv import load_dotenv
import streamlit as st
import chain

load_dotenv()

def poem_generator_app():
    """
    Poem Generator App
    """
    with st.form("Poem_Generator"):
        st.write("## Poem Generator App")
        topic = st.text_input("Enter topic for the poem")
        submitted = st.form_submit_button("submit",type="primary")
        if(submitted):
            response =chain.generate_poem(topic)
            st.info(response)

poem_generator_app()

