from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import logging
import threading
from typing import Any

import requests
import streamlit as st
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

logger = logging.getLogger(__name__)

# App metadata (icons, description, repo link)

APP_ICON = "🌆"
APP_TITLE = "City Intelligence System"
REPO_URL = "https://github.com/Sheharyar-Sarmad/City-Intelligence-System"
APP_DESCRIPTION = (
    "AI-powered City Intelligence System built with Streamlit, LangChain & Groq. "
    "Live weather via OpenWeather, latest news via Tavily, human-in-the-loop "
    "tool approval, and a 3-call Tavily limit per conversation."
)

TOOL_ICONS = {"get_weather": "🌦️", "get_news": "📰"}
USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"

# Page configuration (page_icon is the browser tab icon / favicon)
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="centered",
    menu_items={
        "Get help": REPO_URL,
        "Report a bug": f"{REPO_URL}/issues",
        "About": f"### {APP_ICON} {APP_TITLE}\n\n{APP_DESCRIPTION}\n\n[View on GitHub]({REPO_URL})",
    },
)

# Environment key check
REQUIRED_KEYS = ["GROQ_API_KEY", "OPENWEATHER_API_KEY", "TAVILY_API_KEY"]
missing_keys = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing_keys:
    st.error(f"Missing environment variables: {', '.join(missing_keys)}. Add them to your .env file.")
    st.stop()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Tavily usage limit (per conversation)

MAX_TAVILY_CALLS = 3


class TavilyUsage:
    """Thread-safe counter of Tavily calls per conversation thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def used(self, thread_id: str) -> int:
        with self._lock:
            return self._counts.get(thread_id, 0)

    def try_acquire(self, thread_id: str) -> bool:
        """Reserve one call. Returns False if the limit is already reached."""
        with self._lock:
            count = self._counts.get(thread_id, 0)
            if count >= MAX_TAVILY_CALLS:
                return False
            self._counts[thread_id] = count + 1
            return True


@st.cache_resource(show_spinner=False)
def get_tavily_usage() -> TavilyUsage:
    return TavilyUsage()


TAVILY_USAGE = get_tavily_usage()

# Tools

@tool
def get_weather(city: str) -> str:
    """Get the current weather conditions for a specified city."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip().strip("'\"")
    clean_city = city.strip()

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": clean_city, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        data: dict[str, Any] = response.json()

        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather data.')}"

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"]
        return f"Weather in {clean_city}: {desc.capitalize()}, {temp}°C, Humidity: {humidity}%"
    except Exception:
        return "Error fetching weather: Unable to connect to OpenWeather service."


@tool
def get_news(city: str, config: RunnableConfig) -> str:
    """Get the latest news about a city."""
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")

    if not TAVILY_USAGE.try_acquire(thread_id):
        return (
            f"News search limit reached ({MAX_TAVILY_CALLS} searches per conversation). "
            "Tell the user no more news searches are available in this conversation "
            "and that they can start a new conversation to search again."
        )

    try:
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

    news_list = []
    for r in results:
        title = r.get("title") or "No title"
        url = r.get("url") or ""
        snippet = r.get("content") or ""
        news_list.append(f"- **{title}**\n  {url}\n  {snippet[:120]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# Output Parsing

def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def parse_agent_output(result: dict[str, Any]) -> dict[str, Any]:
    interrupts: list[dict[str, Any]] = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            interrupts.extend(value.get("action_requests", []))

    messages = result.get("messages", [])

    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    turn = messages[last_human + 1:]

    tool_results = {m.tool_call_id: _message_text(m.content) for m in turn if isinstance(m, ToolMessage)}
    tools: list[dict[str, Any]] = []
    for m in turn:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                tools.append({
                    "name": call["name"],
                    "args": call["args"],
                    "result": tool_results.get(call["id"], "Waiting for approval or not executed."),
                })

    answer = ""
    if not interrupts:
        for m in reversed(turn):
            if isinstance(m, AIMessage) and not m.tool_calls:
                answer = _message_text(m.content)
                break

    return {"answer": answer, "interrupts": interrupts, "tools": tools}


# Agent Chains

@st.cache_resource(show_spinner=False)
def build_chains(require_approval: bool):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5, max_tokens=700)

    middleware = []
    if require_approval:
        policy = {"allowed_decisions": ["approve", "reject"]}
        middleware.append(
            HumanInTheLoopMiddleware(interrupt_on={"get_weather": policy, "get_news": policy})
        )

    agent = create_agent(
        model=llm,
        tools=[get_weather, get_news],
        system_prompt=(
            "You are a helpful assistant that provides weather and news information "
            "for cities. Use the tools provided to fetch the latest data. "
            f"The news tool is limited to {MAX_TAVILY_CALLS} searches per conversation, "
            "so only call it when the user actually asks for news, and never call it "
            "more than once per city request."
        ),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )

    prepare_input = RunnableLambda(
        lambda text: {"messages": [{"role": "user", "content": text}]}
    )
    prepare_resume = RunnableLambda(
        lambda decisions: Command(resume={"decisions": decisions})
    )
    parse_output = RunnableLambda(parse_agent_output)

    ask_chain = prepare_input | agent | parse_output
    resume_chain = prepare_resume | agent | parse_output
    return ask_chain, resume_chain


# Session State & Sidebar

def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.approval_round = 0


if "thread_id" not in st.session_state:
    reset_conversation()

with st.sidebar:
    st.header(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_DESCRIPTION)
    st.link_button("⭐ View on GitHub", REPO_URL, use_container_width=True)

    st.divider()
    st.subheader("⚙️ Settings")
    require_approval = st.checkbox("🔒 Require approval before tool calls", value=True)

    if st.session_state.get("last_mode") not in (None, require_approval):
        reset_conversation()
    st.session_state.last_mode = require_approval

    if st.button("🔄 New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.caption(f"🧵 Thread: {st.session_state.thread_id[:8]}")
    st.caption("🌦️ Ask about the weather or 📰 the latest news in any city.")

    st.divider()
    usage_placeholder = st.empty()  # filled at the end of the script so it is always current

ask_chain, resume_chain = build_chains(require_approval)


# Execution Helper

def run_chain(chain, payload: Any) -> None:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        with st.spinner("🤖 Agent processing..."):
            result = chain.invoke(payload, config=config)
    except Exception as exc:
        logger.exception("Agent run failed")
        st.session_state.pending = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"The agent hit an error ({type(exc).__name__}). Try again, or start a new conversation.",
            "tools": [],
        })
        return

    if result["interrupts"]:
        st.session_state.pending = result["interrupts"]
        st.session_state.approval_round += 1
    else:
        st.session_state.pending = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"] or "No response was produced.",
            "tools": result["tools"],
        })


# UI Rendering

def tool_icon(name: str) -> str:
    return TOOL_ICONS.get(name, "🛠️")


def render_tools(tools: list[dict[str, Any]]) -> None:
    if not tools:
        return
    with st.status(f"🛠️ Executed {len(tools)} tool call(s)", state="complete", expanded=False):
        for t in tools:
            st.markdown(f"{tool_icon(t['name'])} **Tool:** `{t['name']}`")
            st.json(t["args"])
            st.code(t["result"], language="text")


def render_tavily_usage() -> None:
    used = TAVILY_USAGE.used(st.session_state.thread_id)
    with usage_placeholder.container():
        st.subheader("📰 Tavily usage")
        st.progress(min(used / MAX_TAVILY_CALLS, 1.0))
        st.caption(f"News searches used: {used}/{MAX_TAVILY_CALLS}")
        if used >= MAX_TAVILY_CALLS:
            st.warning("News search limit reached. Start a new conversation to reset it.")


def render_approval(pending: list[dict[str, Any]]) -> None:
    st.warning("⚠️ The agent requires approval to execute tool calls:")
    round_id = st.session_state.approval_round

    with st.form(f"approval_form_{round_id}"):
        choices = []
        for i, action in enumerate(pending):
            name = action.get("name", "unknown tool")
            st.markdown(f"{tool_icon(name)} **Action:** `{name}`")
            st.json(action.get("args", {}))
            decision = st.radio(
                "Decision", ["Approve", "Reject"],
                key=f"decision_{round_id}_{i}", horizontal=True,
            )
            reason = st.text_input(
                "Reason (if rejected)", key=f"reason_{round_id}_{i}",
            )
            choices.append((decision, reason))
        submitted = st.form_submit_button("✅ Submit Decision")

    if submitted:
        decisions = []
        for decision, reason in choices:
            if decision == "Approve":
                decisions.append({"type": "approve"})
            else:
                decisions.append({
                    "type": "reject",
                    "message": reason.strip() or "The user denied this tool call.",
                })
        st.session_state.pending = None
        run_chain(resume_chain, decisions)


# Main Application

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(f"{APP_DESCRIPTION}  \n🔗 [{REPO_URL}]({REPO_URL})")

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_tools(msg.get("tools", []))

pending = st.session_state.pending
if pending:
    render_approval(pending)

prompt = st.chat_input("🔍 Ask about a city", disabled=bool(pending))
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "tools": []})
    run_chain(ask_chain, prompt)

# Refresh usage display after any run so the count is always current
render_tavily_usage()