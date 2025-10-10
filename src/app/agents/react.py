from src.app.common.models import init_chat_model
from src.app.common.prebuild.agents import create_agent

from langchain_tavily import TavilySearch
from src.app.common.tools.mcp import aget_all_mcp_tools, get_all_mcp_tools

from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from src.app.common.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

tavily_search_tools = [
    TavilySearch(max_results=5),
]

mcp_tools = get_all_mcp_tools()

tools = tavily_search_tools + mcp_tools

#chat_model = init_chat_model(model="claude-sonnet-4-5-20250929", model_provider="anthropic")
chat_model = init_chat_model(model="moonshotai/kimi-k2-instruct-0905", model_provider="groq")

agent = create_agent(
    chat_model,
    tools=tools
)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage


    agent.invoke(
        {"messages": [HumanMessage(content="what is the latest price of Bitcoin?")]}
    )
