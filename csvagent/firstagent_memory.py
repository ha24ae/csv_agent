from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from openai.types.beta.realtime.conversation_created_event import Conversation

# Load environment variables from .env file
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"

##here we create a model
model = ChatAnthropic(api_key=anthropic_api_key, model=llm_name)

#here we create an object for langchain
messages = [
    SystemMessage(
    content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage(content="who was the very first computer scientist?"),
]

def first_agent(conversation):
    res = model.invoke(conversation)
    return res

#second function that will include a loop and take user function
def run_agent():
    print("Simple AI Agent: Type 'exit' to quit")

    conversation = [SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    )]

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye, you have quit")
            break
        print("AI Agent is thinking...")

        conversation.append(HumanMessage(content=user_input))
        print(conversation)

        # messages = [HumanMessage(content=user_input)]
        response = first_agent(conversation)
        conversation.append(AIMessage(content=response.content))


        print("AI Agent: getting the response...")
        print(f"AI Agent: {response.content}")


        print("\n--- Conversation History ---")
        for message in conversation:
            print(f"{message.__class__.__name__}: {message.content}")
        print("----------------------------\n")

if __name__ == "__main__":
    run_agent()


