# 🌆 City Intelligence System

AI-powered City Intelligence System built with **Streamlit**, **LangChain** & **Groq**. Get live weather via OpenWeather 🌦️, the latest news via Tavily 📰, with human-in-the-loop tool approval ✅ and a 3-call Tavily limit per conversation 🔒.

> 🚀 This is the project built after **GenAI Part 3** of the [**ai-zero-to-hero**](https://github.com/Sheharyar-Sarmad/ai-zero-to-hero) series. The lesson it builds on is here: [`23_genai_3`](https://github.com/Sheharyar-Sarmad/ai-zero-to-hero/tree/main/23_genai_3).

---

## 🔗 Links

| | |
|---|---|
| 🌐 **Live Demo** | _Coming soon (will be updated after deployment)_ |
| 💻 **Project Repository** | [City-Intelligence-System](https://github.com/Sheharyar-Sarmad/City-Intelligence-System) |
| 📚 **AI Zero to Hero (full series)** | [ai-zero-to-hero](https://github.com/Sheharyar-Sarmad/ai-zero-to-hero) |
| 📖 **GenAI Part 3 lesson** | [23_genai_3](https://github.com/Sheharyar-Sarmad/ai-zero-to-hero/tree/main/23_genai_3) |
| 👤 **LinkedIn** | [Sheharyar Sarmad](https://www.linkedin.com/in/sheharyar-sarmad-9b7736289/) |
| 🐙 **GitHub** | [Sheharyar-Sarmad](https://github.com/Sheharyar-Sarmad) |

---

## ✨ Features

- 🌦️ **Live weather** for any city through the OpenWeather API
- 📰 **Latest news** for any city through the Tavily API
- ✅ **Human-in-the-loop approval**: approve or reject every tool call before it runs
- 🔒 **Tavily usage limit**: a hard cap of 3 news searches per conversation to protect your API quota
- 📊 **Usage tracker** in the sidebar showing how many news searches are left
- 🧠 **Conversation memory** per thread, with a "New conversation" button to reset
- 💻 **Two interfaces**: a Streamlit web app and a command line version

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** for the web UI
- **LangChain** and **LangGraph** for the agent, middleware and memory
- **Groq** (`openai/gpt-oss-120b`) as the LLM
- **OpenWeather API** for weather data
- **Tavily API** for news search

---

## 📁 Project Structure

```
city_intelligence/
├── app.py              # Streamlit web app
├── cli_base_app.py     # Command line version
├── requirements.txt    # Dependencies (pip)
├── pyproject.toml      # Project config (uv)
├── uv.lock             # Locked dependency versions (uv)
├── .env.example        # Example environment variables
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sheharyar-Sarmad/City-Intelligence-System.git
cd City-Intelligence-System
```

### 2. Install dependencies

Using **uv** (recommended):

```bash
uv sync
```

Or using **pip**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add your API keys

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Get your keys here:

- Groq: https://console.groq.com
- OpenWeather: https://openweathermap.org/api
- Tavily: https://tavily.com

> ⚠️ Never commit your `.env` file. Keep it listed in `.gitignore`.

---

## ▶️ Run the App

**Streamlit web app:**

```bash
streamlit run app.py
```

**Command line version:**

```bash
python cli_base_app.py
```

---

## 💡 Example Questions

- _What's the weather in Lahore?_
- _Show me the latest news in London._
- _Give me the weather and news for Tokyo._

---

## 🔒 How the Safety Features Work

- **Tool approval:** with approval turned on in the sidebar, the agent pauses before each tool call and shows the tool name and arguments. You choose **Approve** or **Reject** (with an optional reason).
- **Tavily limit:** every news search is counted per conversation. After 3 searches, the news tool stops calling Tavily and the agent tells you the limit was reached. Starting a new conversation resets the count.

---

## 🗺️ Roadmap

- [ ] Deploy the app and add the live demo link
- [ ] Add a 5-day weather forecast
- [ ] Support temperature units (°C / °F)
- [ ] Persistent memory across sessions

---

## 👨‍💻 Author

**Sheharyar Sarmad**

- LinkedIn: [sheharyar-sarmad](https://www.linkedin.com/in/sheharyar-sarmad-9b7736289/)
- GitHub: [Sheharyar-Sarmad](https://github.com/Sheharyar-Sarmad)

Part of the [**ai-zero-to-hero**](https://github.com/Sheharyar-Sarmad/ai-zero-to-hero) learning journey. If you found this useful, consider giving the repo a ⭐.
