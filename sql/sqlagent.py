import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from sqlalchemy import create_engine
import pandas as pd

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
llm_name = "claude-3-haiku-20240307"
model = ChatAnthropic(api_key=api_key, model=llm_name, temperature=1, max_tokens=1000)

##libraries that allow handling of SQL database
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

#creating a path for the sql file
#using SQLite instead of connecting to a server, the engine will be a file locally saved

database_file_path = "./db/performance.db"

#creating an engine to connect to SQLite database
engine = create_engine(f"sqlite:///{database_file_path}")
file_url= "../data/Academic_performance_retention_dataset.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("Academic performance retention dataset", con=engine, if_exists="replace")
db= SQLDatabase.from_uri(f"sqlite:///{database_file_path}")

toolkit= SQLDatabaseToolkit(
    db = db,
    llm = model
)

#print(df)
agent = create_sql_agent(
    llm= model,
    toolkit= toolkit,
    top_k= 30,
    agent_executor_kwargs={'handle_parsing_errors':True},
    verbose= True
)

question = "what are the column names."
res= agent.invoke(question)

