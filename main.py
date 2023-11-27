import streamlit as st
import langchain
from langchain.llms import GooglePalm
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import api

# Sidebar for file upload
st.sidebar.title("Upload CSV File")
file = st.sidebar.file_uploader("Choose a file")
if file is not None:
    csv_file = file
else:
    csv_file = None

def generate_response(question, csv_file):
    """Generates a response to a question using the CSV agent."""
    agent = create_csv_agent(
        GooglePalm(temperature=0.5, google_api_key=api.api_key),
        csv_file,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    response = agent.run(question)
    return response

# Main area for input and display
st.title("Ask me anything about your data!")
question = st.text_input("Enter your question:")

if question:
    response = generate_response(question, csv_file)
    st.write("Response:")
    st.info(response)
else:
    st.write("Please enter a question to get a response.")

