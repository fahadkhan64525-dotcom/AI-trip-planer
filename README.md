# ✈️ Agentic AI Travel Planning Assistant

An intelligent travel planning system built with **LangChain tools**, **OpenAI**, and **Streamlit** that creates personalized trip itineraries for Indian destinations.

---

## 🎯 Project Overview

This project solves the problem of fragmented travel planning by creating a single assistant that:
- Searches flights from structured JSON datasets
- Recommends hotels based on budget and preferences
- Discovers tourist attractions and plans day-wise itineraries
- Fetches real-time weather forecasts (via Open-Meteo API — no key needed)
- Calculates a complete budget breakdown
- Can run fully in local demo mode, with optional OpenAI polish when an API key is available

---

## 📁 Project Structure

```
travel_agent/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
│
├── agent/
│   ├── __init__.py
│   └── travel_agent.py       # LangChain ToolCalling Agent
│
├── tools/
│   ├── __init__.py
│   └── travel_tools.py       # 5 LangChain @tool functions
│
├── utils/
│   ├── __init__.py
│   └── helpers.py            # Query parsing, formatting utilities
│
└── data/
    ├── flights.json           # Flight dataset (20 routes)
    ├── hotels.json            # Hotel dataset (20 properties)
    └── places.json            # Places/POI dataset (28 attractions)
```

---

## 🚀 Setup Instructions

### 1. Clone / Download the Project
```bash
cd travel_agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
An **OpenAI API key** is optional (get one at [platform.openai.com](https://platform.openai.com)).

Either:
- Enter it in the Streamlit sidebar at runtime, or
- Create a `.env` file:
  ```
  OPENAI_API_KEY=sk-your-key-here
  ```

If no key is provided, the app still runs using the local planner and bundled datasets.

### 4. Run the Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 How the Planner Works

The system gathers trip data through the project tools first, then optionally asks OpenAI to turn those results into a richer final itinerary:

```
User Query
    ↓
Local Planner
    ├── search_flights("source:Delhi destination:Goa preference:cheapest")
    ├── recommend_hotels("city:Goa budget:5000 preference:rating")
    ├── discover_places("city:Goa type:All days:3")
    ├── get_weather("city:Goa days:3")
    ├── estimate_budget("flight:3900 hotel:3200 days:3 travelers:2 style:moderate")
    ↓
Deterministic itinerary (always available)
    ↓
Optional ChatOpenAI rewrite
    ↓
Final Structured Itinerary
```

---

## 🛠️ LangChain Tools

| Tool | Description | Data Source |
|------|-------------|-------------|
| `search_flights` | Find cheapest/fastest flights | `flights.json` |
| `recommend_hotels` | Top-rated hotels by city & budget | `hotels.json` |
| `discover_places` | Day-wise attraction planning | `places.json` |
| `get_weather` | Live weather forecast | Open-Meteo API |
| `estimate_budget` | Full trip cost breakdown | Computed |

---

## 📊 Sample Output

```
═══════════════════════════════════════════════════
🌟 YOUR 3-DAY TRIP TO GOA (May 28 – May 30, 2026)
═══════════════════════════════════════════════════

✈️ FLIGHT SELECTED:
   SpiceJet | ₹3,900 | Departs 14:00 → Goa

🏨 HOTEL BOOKED:
   Sea View Resort | ₹3,200/night | ⭐ 4.5 | Pool, Beach Access

🌤️ WEATHER FORECAST:
   Day 1: ☀️ Clear Sky | High: 32°C
   Day 2: ⛅ Partly Cloudy | High: 31°C
   Day 3: ☀️ Clear Sky | High: 33°C

📅 ITINERARY:
   Day 1: Baga Beach, Calangute Beach
   Day 2: Basilica of Bom Jesus, Old Goa Heritage Walk
   Day 3: Water Sports at Calangute, Anjuna Flea Market

💰 BUDGET BREAKDOWN:
   ✈️ Flights       : ₹  7,800
   🏨 Hotel         : ₹  9,600
   🍽️ Food          : ₹  7,200
   🚕 Local Travel  : ₹  1,800
   🎭 Activities    : ₹  3,000
   🛍️ Miscellaneous : ₹  2,400
   ─────────────────────────────
   💵 TOTAL         : ₹ 31,800
   👤 Per Person    : ₹ 15,900
```

---

## 🌐 APIs Used

- **OpenAI GPT-4o-mini** — Optional itinerary polishing
- **Open-Meteo API** — Free real-time weather (no API key required)
  - Endpoint: `https://api.open-meteo.com/v1/forecast`

---

## 📖 Key Technologies

- **LangChain** — Tool wrappers and optional model integration
- **OpenAI** — LLM backbone for reasoning
- **Streamlit** — Interactive web interface
- **Python** — Core language (PEP 8 compliant)

---

## 📝 Code Quality

- ✅ Modular architecture (tools / agent / utils separated)
- ✅ Docstrings on all functions and modules
- ✅ Error handling with try-except throughout
- ✅ PEP 8 compliant formatting
- ✅ Descriptive variable and function names
- ✅ No hardcoded credentials

---

## 🔗 References

- [LangChain Docs](https://docs.langchain.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Open-Meteo API](https://open-meteo.com/)
- [OpenAI Platform](https://platform.openai.com/)
