from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic


# Load environment variables from .env file
load_dotenv()


anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

llm_name = "claude-3-haiku-20240307"

##here we create a model
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name)

##here we create an object for langchain
messages = [
    SystemMessage(
    content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="who was the very first computer scientist?"),
]

##if we want to run it we pass the messages to the model
res = model.invoke(messages)
print(res)

#The example above is hard coded
#What is we want to let the user input their own question

# First function
def first_agent(messages):
    res = model.invoke(messages)
    return res

# second function that will include a loop and take user function
def run_agent():
    print("Simple AI Agent: Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye, you have quit")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent: getting the response...")
        print(f"AI Agent: {response.content}")

if __name__ == "__main__":
    run_agent()


