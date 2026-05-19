from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from langchain.agents import create_agent
# from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

# @tool
# def search(query: str) -> str:
#     """
#         Tool that searches over internet
#         args:
#             query: the query to search for
#         returns:
#             the search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer"""
    answer:str = Field(description="The answer to the query")
    sources: list[Source] = Field(default_factory=list, description="The list of sources that are used to generate answer")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from react-search-agent!")
    result = agent.invoke({"messages": HumanMessage(content = "search for 3 jobs for full stack developer using langchain in the pune, India and list their details")})
    print(result)

if __name__ == "__main__":
    main()
