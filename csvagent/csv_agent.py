from idlelib.query import Query
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langgraph.checkpoint.memory import MemorySaver


# Load environment variables from .env file
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name, temperature=1, max_tokens=1000)

#We want to read the csv file
df= pd.read_csv("./data/Academic_performance_retention_dataset.csv").fillna(value=0)
#print(df.head())

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
    create_csv_agent,
)

agent = create_pandas_dataframe_agent(
    llm= model,
    df= df,
    verbose=True,
    #this is the key argument that unlocks Python REPL calls
    allow_dangerous_code=True, #if you have the latest version of langchain you dont need this
)

# response = agent.invoke(["How many columns is in this?"])
# print(response)

#we are going to inject these prompts
CSV_PROMPT_PREFIX = """
    First set the pandas display options to show all the columns,
    get the column names,
    then answer the question
    You are working with a pandas dataframe in Python. 
"""

CSV_PROMPT_SUFFIX ="""
    - **ALWAYS** before giving the Final Answer, try another method.
    Then reflect on the answers of the two methods you did and ask yourself?
    if it correctly answers the original question.
    If you are not sure, try another method.
    - If the methods tried do not give the same result, reflect and try again until you have two methods that have the same result.
    - If you still can not arrive to a consistent result, say that you are not sure of the answer.
    - If you are sure of the correct answer, creat a beautifuland thorough response using Markdown.
    - ***DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
    only use the results of the CALCULATIONS YOU HAVE DONE**.
    - **ALWAYS**, as part of your "Final Answer", explain how you got to the answer on a section that starts with:"\n\nExplanation:\n".
    In the explanation, mention the column names that you used to get the final answer.
"""

# question = "What is the mean age of students?"
# response = agent.invoke(CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX)
# print(response)

##note
##Without the prompt when asking the agent what is the mean age of students I got an error
##With the prompt we got an answer

# def run_csv_agent():
#     print("Simple CSV query AI Agent: Type 'exit' to quit")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye, you have quit")
#             break
#         print("AI Agent is thinking...")
#         response = agent.invoke(CSV_PROMPT_PREFIX + user_input + CSV_PROMPT_SUFFIX)
#         print("AI Agent: getting the response...")
#         print(f"AI Agent: {response}")
#
# if __name__ == "__main__":
#     run_csv_agent()

# def run_agent():
#     print("Simple AI Agent: Type 'exit' to quit")
#
#     conversation = []
#
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == "exit":
#             print("Goodbye, you have quit")
#             break
#         print("AI Agent is thinking...")
#
#         conversation.append(HumanMessage(content=user_input))
#         print(conversation)
#         question = CSV_PROMPT_PREFIX + user_input + CSV_PROMPT_SUFFIX
#         # messages = [HumanMessage(content=user_input)]
#         response = csv_agent(conversation)
#         conversation.append(AIMessage(content=response.content))
#
#
#         print("AI Agent: getting the response...")
#         print(f"AI Agent: {response.content}")
#
#
#         print("\n--- Conversation History ---")
#         for message in conversation:
#             print(f"{message.__class__.__name__}: {message.content}")
#         print("----------------------------\n")
#
# if __name__ == "__main__":
#     run_agent()


import streamlit as st

st.title("CSV Agent")
st.write("### Dataset Preview")
st.write(df.head())

# ##Lets allow the user to write their own question
st.write("### Ask a Question on this dataset")
question = st.text_input(
    "Enter you question about the dataset:"
)

#we need to creat a button for the search query
if st.button("Run Query"):
    query = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    response = agent.invoke(query)
    st.write("Final Answer")
    st.markdown(response["output"])

