from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
import pandas as pd

# Load environment variables from .env file
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name)

#We want to read the csv file
df= pd.read_csv("../data/Academic performance retention dataset.csv").fillna(value=0)
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
    allow_dangerous_code=True #if you have the latest version of langchain you dont need this
)

# response = agent.invoke("How many does it have?")
# print(response)

#we are going to inject these prompts
CSV_PROMPT_PREFIX = """
    First set the pandas display options to show all the columns,
    get the column names,
    then answer the question
"""
#by asking it to use two methods we are making sure the model is not hallucinating
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

question = "What is the mean age of students?"
response = agent.invoke(CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX)
print(response)

##Without the prompt when asking the agent what is the mean age of students I got an error
##With the prompt we got an answer