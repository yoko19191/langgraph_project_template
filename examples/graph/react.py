from src.app.common.prebuild.agents import create_agent
from src.app.common.models import init_chat_model

from langchain_tavily import TavilySearch, TavilyMap, TavilyCrawl, TavilyExtract

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

tavily_search = TavilySearch(max_results=5)
chat_model = init_chat_model(model="claude-sonnet-4-5-20250929", model_provider="anthropic")
react_agent = create_agent(
    chat_model,
    tools=[tavily_search]
)