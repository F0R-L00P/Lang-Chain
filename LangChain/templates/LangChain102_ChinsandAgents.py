# langchain core packages
import os
from dotenv import load_dotenv

import transformers
import langchain_huggingface
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
load_dotenv(dotenv_path=dotenv_path, override=True)

API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
# -----------------------------------------------------------
# --------------------- Sequential Chains -------------------
# -----------------------------------------------------------
destination_prompt = PromptTemplate(
    input_variables=["destination"],
    template="I am planning a trip to {destination}. Can you suggest some activities to do there?",
)

activities_prompt = PromptTemplate(
    input_variables=["activities"],
    template="I only have one day, so can you create an itinerary from your top three activities {activities}.",
)

llm2 = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100},
)

# Run the pipeline destination_prompt | llm | StrOutputParser() using the current input.
# Take its output (a string).
# Put that string into a new dictionary under the key "activities".
sequential_chain = (
    {"activities": destination_prompt | llm2 | StrOutputParser()}
    | activities_prompt
    | llm2
    | StrOutputParser()
)

print(sequential_chain.invoke({"destination": "Paris"}))

# -----------------------------------------------------------
# ----------------------- Agents ----------------------------
# -----------------------------------------------------------
from langchain_ibm import ChatWatsonx
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.agent_toolkits.load_tools import load_tools


llm = ChatWatsonx(
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    url=ENDPOINT,
    project_id=PROJECT_ID,
    api_key=API_KEY,
    params={"max_new_tokens": 500, "decoding_method": "greedy"},
)


tools = load_tools(["llm-math"], llm=llm)
agent = create_agent(model=llm, tools=tools)

question = "Compute sqrt(101) to 20 decimal places"  # "compute the square root of 101?

result = agent.invoke({"messages": [HumanMessage(content=question)]})

print(result["messages"][-1].content)
# -----------------------------------------------------------
# ----------------------- Tools -----------------------------
# -----------------------------------------------------------
# tools must be compatible and accassible via name attribute
print(tools[0].name)  # name of the tool
print(tools[0].description)  # llm decides what the tool does to use
print(tools[0].return_direct)  # wehther the agent should stop after invoking the tool
# @tool will make the function a tool method for the model to use
# ensure the description is verbose
