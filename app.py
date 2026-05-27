"""
app.py
-------
Streamlit UI for the Agentic AI Travel Planning Assistant.
Provides an interactive interface for trip planning with real-time
agent reasoning display and structured itinerary output.
"""

import sys
import os
import html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from utils.helpers import (
    extract_trip_details,
    build_agent_query,
    format_steps_for_display,
    validate_api_key,
    get_destination_tips,
)

load_dotenv()

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        color: #f0e6d3;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a8c0cc;
        font-size: 1.05rem;
        margin: 0.5rem 0 0;
    }

    .destination-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.4rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #e0e0e0;
    }

    .destination-card:hover {
        border-color: #e94560;
        transform: translateY(-2px);
    }

    .step-card {
        background: #f8f9fa;
        border-left: 4px solid #2c5364;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    .result-box {
        background: linear-gradient(180deg, #fbfdff 0%, #eef5fb 100%);
        border: 1px solid #cfdae5;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin: 0;
        overflow-x: auto;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        line-height: 1.7;
        color: #10263d !important;
        box-shadow: 0 10px 28px rgba(16, 38, 61, 0.08);
        white-space: pre-wrap;
        font-weight: 500;
    }

    .result-box,
    .result-box * {
        color: #12263a !important;
    }

    .tip-badge {
        background: #e8f4f8;
        border: 1px solid #b3d9e8;
        border-radius: 20px;
        padding: 0.4rem 0.9rem;
        font-size: 0.85rem;
        color: #2c5364;
        display: inline-block;
        margin: 0.2rem;
    }

    .stTextArea textarea {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e1e8ef;
        border-radius: 10px;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ AI Travel Planning Assistant</h1>
    <p>Powered by LangChain · Agentic AI · Real-time Weather · Smart Itineraries</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input(
        "OpenAI or Hugging Face API Key",
        type="password",
        placeholder="sk-... or hf_...",
        help="Optional. Add a key for an AI-polished itinerary (OpenAI or Hugging Face), or leave blank to use local demo mode."
    )
    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    env_hf_token = os.getenv("HF_TOKEN", "").strip()
    effective_api_key = api_key.strip() or env_api_key or env_hf_token
    if env_api_key and not api_key.strip():
        st.caption("Using `OPENAI_API_KEY` from your environment.")
    elif env_hf_token and not api_key.strip() and not env_api_key:
        st.caption("Using `HF_TOKEN` from your environment.")
    elif not effective_api_key:
        st.caption("No API key detected. The app will use local demo mode.")

    st.divider()

    st.markdown("### ⚡ Quick Presets")
    preset = st.radio(
        "Select a trip template:",
        options=[
            "🏖️ Beach Getaway — Goa (3 days)",
            "🌿 Nature & Culture — Kerala (5 days)",
            "🏰 Heritage Tour — Rajasthan (4 days)",
            "❄️ Mountain Adventure — Manali (4 days)",
            "🏙️ City Break — Mumbai (2 days)",
        ],
        index=0
    )

    st.divider()

    st.markdown("### 📊 Trip Style")
    travel_style = st.select_slider(
        "Budget Style",
        options=["Budget", "Moderate", "Luxury"],
        value="Moderate"
    )

    num_travelers = st.number_input("👥 Number of Travelers", min_value=1, max_value=10, value=2)

    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        "This assistant uses LangChain ReAct agents to autonomously "
        "search flights, hotels, attractions, and live weather — "
        "then generates a complete personalized itinerary."
    )


# ─── Preset → Query Mapping ────────────────────────────────────────────────────
PRESET_QUERIES = {
    "🏖️ Beach Getaway — Goa (3 days)":
        f"Plan a 3-day moderate beach trip to Goa for {num_travelers} travelers from Delhi.",
    "🌿 Nature & Culture — Kerala (5 days)":
        f"Plan a 5-day moderate nature trip to Kerala for {num_travelers} travelers from Delhi.",
    "🏰 Heritage Tour — Rajasthan (4 days)":
        f"Plan a 4-day moderate heritage trip to Rajasthan for {num_travelers} travelers from Delhi.",
    "❄️ Mountain Adventure — Manali (4 days)":
        f"Plan a 4-day adventure trip to Manali for {num_travelers} travelers from Delhi.",
    "🏙️ City Break — Mumbai (2 days)":
        f"Plan a 2-day city trip to Mumbai for {num_travelers} travelers from Delhi.",
}


# ─── Main Input Area ───────────────────────────────────────────────────────────
st.markdown("### 💬 Describe Your Dream Trip")

col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_area(
        "Tell the AI what you're looking for:",
        value=PRESET_QUERIES.get(preset, ""),
        height=110,
        placeholder="e.g. Plan a 4-day romantic trip to Goa for 2 people from Delhi with a budget of ₹25,000...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    plan_btn = st.button("🚀 Plan My Trip", use_container_width=True, type="primary")
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("result", None)
    st.session_state.pop("steps", None)
    st.rerun()


# ─── Parse & Preview Trip Details ─────────────────────────────────────────────
if user_query:
    details = extract_trip_details(user_query)
    details["travelers"] = num_travelers
    details["style"] = travel_style.lower()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📍 Destination", details.get("destination") or "—")
    with c2:
        st.metric("🛫 From", details.get("source", "Delhi"))
    with c3:
        st.metric("📅 Duration", f"{details.get('days', 3)} days")
    with c4:
        st.metric("👥 Travelers", details.get("travelers", 1))


# ─── Run Agent ─────────────────────────────────────────────────────────────────
if plan_btn:
    if not user_query.strip():
        st.warning("Please describe your trip first!")
    elif api_key.strip() and not validate_api_key(api_key):
        st.error("⚠️ Please enter a valid OpenAI API key in the sidebar.")
    else:
        details = extract_trip_details(user_query)
        details["travelers"] = num_travelers
        details["style"] = travel_style.lower()
        agent_query = build_agent_query(details)

        with st.spinner("🤖 AI Agent is planning your trip..."):
            progress = st.progress(0, text="Initializing agent...")

            try:
                # Dynamic import to avoid errors when key is missing
                from agent.travel_agent import run_travel_agent

                progress.progress(20, text="🧭 Gathering flights, stays, and activities...")
                result = run_travel_agent(
                    agent_query,
                    effective_api_key,
                    trip_details=details,
                )

                if result.get("mode") == "ai":
                    progress.progress(100, text="✅ AI itinerary ready!")
                else:
                    progress.progress(100, text="✅ Demo itinerary ready!")
                progress.empty()

                st.session_state["result"]      = result["output"]
                st.session_state["steps"]       = result.get("steps", [])
                st.session_state["destination"] = details.get("destination", "")
                st.session_state["planner_mode"] = result.get("mode", "demo")

            except ImportError:
                progress.empty()
                st.error("❌ Required packages are missing. Run: pip install -r requirements.txt")
            except Exception as e:
                progress.empty()
                st.error(f"❌ Agent error: {str(e)}")
                st.info("💡 Common issues: invalid API key, rate limit exceeded, or blocked network access.")


# ─── Display Results ───────────────────────────────────────────────────────────
if "result" in st.session_state:
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🗺️ Your Itinerary", "🔍 Agent Reasoning", "💡 Travel Tips"])

    with tab1:
        st.markdown("### 🌟 Your Personalized Travel Plan")
        if st.session_state.get("planner_mode") == "demo":
            st.info("Demo mode used the local datasets and offline planner. Add an API key to get an AI-polished itinerary.")
        st.markdown(
            f'<pre class="result-box">{html.escape(st.session_state["result"])}</pre>',
            unsafe_allow_html=True
        )

        # Download button
        st.download_button(
            label="📄 Download Itinerary (TXT)",
            data=st.session_state["result"],
            file_name=f"trip_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with tab2:
        st.markdown("### 🔍 How the Agent Planned Your Trip")
        steps = format_steps_for_display(st.session_state.get("steps", []))

        if not steps:
            st.info("No intermediate steps captured. The agent may have responded directly.")
        else:
            for i, step in enumerate(steps, 1):
                with st.expander(f"{step['icon']} Step {i}: {step['tool'].replace('_', ' ').title()}"):
                    st.markdown("**🔧 Tool Input:**")
                    st.code(step["input"], language="text")
                    st.markdown("**📤 Tool Output:**")
                    st.text(step["output"])

    with tab3:
        destination = st.session_state.get("destination", "")
        tips = get_destination_tips(destination)
        st.markdown(f"### 💡 Tips for Visiting {destination or 'Your Destination'}")
        for tip in tips:
            st.markdown(f'<span class="tip-badge">{tip}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 General Travel Checklist")
        checklist_items = [
            "✅ Carry government-issued photo ID",
            "✅ Download Google Maps offline for the region",
            "✅ Check if your destination needs e-permits",
            "✅ Inform your bank about travel to avoid card blocks",
            "✅ Keep emergency contacts saved offline",
            "✅ Travel insurance is highly recommended",
        ]
        for item in checklist_items:
            st.markdown(item)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built with ❤️ using LangChain · OpenAI · Streamlit · Open-Meteo API</small></center>",
    unsafe_allow_html=True
)
