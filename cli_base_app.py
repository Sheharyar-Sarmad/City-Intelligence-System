# City Intelligence System
#
# A command line agent that answers questions about the weather and the latest
# news for a city. It uses a Groq-hosted LLM, the OpenWeather API for weather,
# the Tavily API for news, and a human approval step before every tool call.

# Load environment variables from a .env file before anything reads them.
# Required keys: GROQ_API_KEY, OPENWEATHER_API_KEY, TAVILY_API_KEY
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from typing import Any, TypedDict, Callable
from functools import lru_cache
from rich import print

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest, wrap_tool_call
from langgraph.types import Command


# Shape of the value returned by agent.invoke(). The agent returns a dict
# whose "messages" key holds the full conversation, including tool calls.
class AgentResult(TypedDict):
    messages: list[BaseMessage]


# Tool 1: weather

# The @tool decorator turns the function into a tool the LLM can call.
# The LLM reads the function name, the type hints, and the docstring to decide
# when to call it and what arguments to pass, so the docstring matters.
#
# lru_cache remembers results per city for the lifetime of the program, so
# asking about the same city twice does not call the API again. Be aware that
# this also means the weather never refreshes until the program restarts, and
# error strings are cached too. Remove it if you need live data.
@tool
@lru_cache
def get_weather(city: str) -> str:
    """Get the current weather conditions for a specified city."""
    # Clean the key in case the .env value has spaces or surrounding quotes.
    raw_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    api_key: str = raw_key.strip().strip("'\"")
    clean_city: str = city.strip()

    if not api_key:
        return "Error: OpenWeather API key is not configured in environment."

    try:
        # Passing query values through params lets requests URL-encode them,
        # so city names with spaces or special characters work correctly.
        response: requests.Response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": clean_city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        data: dict[str, Any] = response.json()

        # OpenWeather returns "cod" as 200 on success. On failure it returns
        # an error code (for example 404 for an unknown city) and a message.
        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather data.')}"

        temp: float = data["main"]["temp"]
        humidity: int = data["main"]["humidity"]
        desc: str = data["weather"][0]["description"]
        return f"Weather in {clean_city}: {desc.capitalize()}, {temp}°C, Humidity: {humidity}%"

    except Exception:
        # Return a generic message instead of the raw exception, because the
        # exception text can contain the request URL and leak the API key.
        return "Error fetching weather: Unable to connect to OpenWeather service."


# Tool 2: news

# Fail early with a clear message if the Tavily key is missing, instead of
# failing later in the middle of a conversation.
tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

# One shared client is reused by every news request.
tavily_client: TavilyClient = TavilyClient(api_key=tavily_api_key)

@tool
@lru_cache
def get_news(city: str) -> str:
    """Get the latest news about a city."""
    try:
        # "basic" search depth is faster and cheaper than "advanced".
        # max_results limits how many articles come back.
        response: dict[str, Any] = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3,
        )
    except Exception:
        return "Error fetching news: Unable to connect to the news service."

    results: list[dict[str, Any]] = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    # Build a short, readable summary of each article for the LLM.
    news_list: list[str] = []
    for r in results:
        title: str = r.get("title", "No title")
        url: str = r.get("url", "")
        snippet: str = r.get("content", "")
        # Only the first 100 characters of the snippet are kept to save tokens.
        news_list.append(f"- {title}\n  {url}\n  {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# Human in the loop middleware

# @wrap_tool_call turns this function into a middleware object. It does not run
# by itself. It runs automatically around every tool call, but only after it is
# registered with create_agent(..., middleware=[human_approval]) below.
#
# When the LLM decides to call a tool, the agent does not run the tool
# directly. It first passes the call through this function:
#   request  holds the tool call (name, arguments, id) the LLM wants to make.
#   handler  is a function that actually runs the tool (or the next middleware).
# Calling handler(request) executes the tool. Not calling it blocks the tool.
@wrap_tool_call
def human_approval(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
) -> ToolMessage | Command[Any]:
    """Ask for human approval before every tool call."""
    tool_name: str = str(request.tool_call["name"])
    tool_args: Any = request.tool_call.get("args", {})

    confirm: str = input(
        f"Agent wants to call '{tool_name}' with {tool_args}. Approve? (yes/no): "
    )

    # Anything other than "yes" denies the call. The tool is never executed.
    # A ToolMessage must be returned so the LLM receives a response for its
    # tool call id and can tell the user that the action was denied.
    if confirm.strip().lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=str(request.tool_call["id"]),
        )

    # Approved: run the real tool and return its result to the agent.
    return handler(request)


# Model and agent

# temperature controls randomness (lower is more predictable).
# max_tokens limits the length of each model response.
llm: ChatGroq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=700,
)

# create_agent builds the loop: the LLM reads the user message, decides whether
# to call a tool, the tool result goes back to the LLM, and this repeats until
# the LLM produces a final answer.
agent = create_agent(
    model=llm,
    tools=[get_weather, get_news],
    system_prompt=(
        "You are a helpful assistant that provides weather and news information "
        "for cities. Use the tools provided to fetch the latest data."
    ),
    # Registering the middleware here is what makes it run on every tool call.
    # Without this line the approval prompt is never shown.
    middleware=[human_approval],
)


# Command line loop
def main() -> None:
    print("\nCity Intelligence System | Type 'exit' to quit\n")

    while True:
        user_input: str = input("\nUser: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\nExiting the City Intelligence System. Goodbye!\n")
            break

        # Each call starts a fresh conversation, so the agent does not remember
        # earlier questions. Add a checkpointer to keep memory between turns.
        result: AgentResult = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        # The last message in the list is the agent's final answer.
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()