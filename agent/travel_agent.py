"""
agent/travel_agent.py
---------------------
Core planner for the travel assistant.

The planner always gathers data locally through the project tools so the app
remains usable offline. When an OpenAI API key is available, the collected
results are optionally polished into a richer itinerary with ChatOpenAI.
"""

from __future__ import annotations

import re
from typing import Any

from tools.travel_tools import (
    calculate_budget_breakdown,
    discover_places,
    estimate_budget,
    find_flights,
    find_hotels,
    get_weather,
    recommend_hotels,
    search_flights,
    select_places,
)
from utils.helpers import extract_trip_details, get_destination_tips

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Travel Planning Assistant for Indian destinations.
You will receive a user's trip request plus tool outputs that were already gathered.

Write a polished, practical itinerary in this structure:

═══════════════════════════════════════════════════════
🌟 YOUR [N]-DAY TRIP TO [DESTINATION] ([DATE RANGE])
═══════════════════════════════════════════════════════

✈️ FLIGHT SELECTED:
   [Airline] | ₹[Price] | Departs [Time] → [Destination]

🏨 HOTEL BOOKED:
   [Hotel Name] | ₹[Price]/night | [Rating]★ | [Key Amenities]

🌤️ WEATHER FORECAST:
   [Concise day-by-day weather summary]

📅 DAY-WISE ITINERARY:
   Day 1: [Places]
   Day 2: [Places]

💰 BUDGET BREAKDOWN:
   [Clear cost breakdown]

💡 TRAVEL TIPS:
   [2-3 helpful tips]

Use only the provided facts. Do not invent flights, hotels, places, or prices.
Keep the tone enthusiastic and useful.
Return plain text only. Do not use Markdown code fences, tables, or block quotes.
"""


def _merge_details(query: str, trip_details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge parsed details with any explicit details passed by the caller."""
    details = extract_trip_details(query)
    for key, value in (trip_details or {}).items():
        if value not in (None, ""):
            details[key] = value

    details["destination"] = details.get("destination") or "Goa"
    details["source"] = details.get("source") or "Delhi"
    details["days"] = max(int(details.get("days", 3) or 3), 1)
    details["travelers"] = max(int(details.get("travelers", 1) or 1), 1)
    details["style"] = str(details.get("style", "moderate") or "moderate").lower()
    return details


def _extract_date_range(query: str) -> str:
    """Extract the `Trip dates:` segment from the normalized planner query."""
    match = re.search(r"Trip dates:\s*([^\.]+)", query)
    return match.group(1).strip() if match else "Flexible Dates"


def _infer_place_type(query: str) -> str:
    """Infer a preferred place category from the trip request."""
    keywords = (
        ("Beach", ("beach", "coast", "island")),
        ("Adventure", ("adventure", "trek", "sports", "ski", "snow")),
        ("Nature", ("nature", "waterfall", "tea garden", "backwater")),
        ("Heritage", ("heritage", "fort", "palace", "history")),
        ("Culture", ("culture", "museum", "walk", "festival")),
        ("Shopping", ("shopping", "market", "bazaar")),
        ("Spiritual", ("spiritual", "temple", "pilgrim")),
    )
    query_lower = query.lower()
    for place_type, terms in keywords:
        if any(term in query_lower for term in terms):
            return place_type
    return "All"


def _select_flight_preference(style: str) -> str:
    """Choose a flight preference based on travel style."""
    if style == "budget":
        return "cheapest"
    if style == "luxury":
        return "fastest"
    return "best"


def _select_hotel_preference(style: str) -> str:
    """Choose a hotel preference based on travel style."""
    if style == "budget":
        return "cheapest"
    if style == "luxury":
        return "luxury"
    return "rating"


def _estimate_hotel_budget(total_budget: int | None, days: int, style: str) -> int | None:
    """Estimate a nightly hotel budget from the user's overall budget."""
    if not total_budget:
        return None

    share_by_style = {
        "budget": 0.25,
        "moderate": 0.35,
        "luxury": 0.45,
    }
    budget_share = share_by_style.get(style, 0.35)
    return max(int(total_budget * budget_share / max(days, 1)), 800)


def _step(tool: str, tool_input: str, tool_output: str) -> dict[str, str]:
    """Create a normalized reasoning step payload for the UI."""
    return {
        "tool": tool,
        "input": tool_input,
        "output": tool_output,
    }


def _format_default_itinerary(
    details: dict[str, Any],
    date_range: str,
    selected_flight: dict[str, Any] | None,
    selected_hotel: dict[str, Any] | None,
    places_by_day: list[list[dict[str, Any]]],
    weather_text: str,
    budget_text: str,
) -> str:
    """Build a deterministic itinerary that works without a live LLM call."""
    destination = details["destination"].upper()
    days = details["days"]
    tips = get_destination_tips(details["destination"])

    lines = [
        "═══════════════════════════════════════════════════════",
        f"🌟 YOUR {days}-DAY TRIP TO {destination} ({date_range})",
        "═══════════════════════════════════════════════════════",
        "",
        "✈️ FLIGHT SELECTED:",
    ]

    if selected_flight:
        lines.append(
            f"   {selected_flight['airline']} | ₹{selected_flight['price']:,} | "
            f"Departs {selected_flight['departure']} → {selected_flight['destination']}"
        )
    else:
        lines.append("   No matching flight was found in the local dataset.")

    lines.extend(["", "🏨 HOTEL BOOKED:"])
    if selected_hotel:
        amenities = ", ".join(selected_hotel["amenities"][:4])
        lines.append(
            f"   {selected_hotel['name']} | ₹{selected_hotel['price_per_night']:,}/night | "
            f"{selected_hotel['rating']}★ | {amenities}"
        )
    else:
        lines.append("   No matching hotel was found in the local dataset.")

    lines.extend(["", "🌤️ WEATHER FORECAST:"])
    weather_lines = [line for line in weather_text.splitlines() if line.strip()]
    if len(weather_lines) > 1:
        for line in weather_lines[1:]:
            lines.append(f"   {line.strip()}")
    elif weather_lines:
        lines.append(f"   {weather_lines[0]}")
    else:
        lines.append("   Weather information is currently unavailable.")

    lines.extend(["", "📅 DAY-WISE ITINERARY:"])
    for day_index in range(days):
        day_places = places_by_day[day_index] if day_index < len(places_by_day) else []
        if day_places:
            names = ", ".join(place["name"] for place in day_places)
            lines.append(f"   Day {day_index + 1}: {names}")
        else:
            lines.append(f"   Day {day_index + 1}: Free exploration, cafe hopping, and local downtime")

    lines.extend(["", "💰 BUDGET BREAKDOWN:"])
    for line in budget_text.splitlines()[1:]:
        lines.append(f"   {line}" if line.strip() else "")

    lines.extend(["", "💡 TRAVEL TIPS:"])
    for tip in tips[:3]:
        lines.append(f"   {tip}")

    return "\n".join(lines)


def _build_llm_prompt(
    query: str,
    details: dict[str, Any],
    steps: list[dict[str, str]],
) -> str:
    """Create the prompt for the optional AI-polished itinerary."""
    tool_dump = "\n\n".join(
        f"{step['tool']} input:\n{step['input']}\n\n{step['tool']} output:\n{step['output']}"
        for step in steps
    )
    return (
        f"User request:\n{query}\n\n"
        f"Trip details:\n{details}\n\n"
        f"Collected tool results:\n{tool_dump}\n"
    )


def _sanitize_itinerary_output(text: str) -> str:
    """Normalize model output so it renders as readable plain text in the UI."""
    cleaned_lines: list[str] = []

    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```") or stripped == "~~~":
            continue

        # Avoid accidental markdown code blocks from 4+ leading spaces.
        leading_spaces = len(line) - len(line.lstrip(" "))
        if leading_spaces >= 4:
            line = f"  {line.lstrip()}"

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text


def _generate_ai_itinerary(
    query: str,
    api_key: str,
    details: dict[str, Any],
    steps: list[dict[str, str]],
) -> str | None:
    """Use ChatOpenAI to rewrite the gathered facts into a polished itinerary."""
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    llm_kwargs = {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "api_key": api_key,
    }
    if api_key.startswith("hf_"):
        llm_kwargs["base_url"] = "https://router.huggingface.co/v1"

    llm = ChatOpenAI(**llm_kwargs)

    try:
        response = llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", _build_llm_prompt(query, details, steps)),
        ])
    except Exception:
        return None

    content = getattr(response, "content", "")
    if isinstance(content, list):
        combined = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()
        return _sanitize_itinerary_output(combined) or None
    return _sanitize_itinerary_output(str(content)) or None


def run_travel_agent(
    query: str,
    api_key: str | None = None,
    trip_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute the travel planner.

    The planner always gathers tool outputs from local datasets and optional
    live weather. If an API key is available and the OpenAI call succeeds,
    the final itinerary is rewritten by the model; otherwise a deterministic
    itinerary is returned.
    """
    details = _merge_details(query, trip_details=trip_details)
    date_range = _extract_date_range(query)
    place_type = _infer_place_type(query)
    flight_preference = _select_flight_preference(details["style"])
    hotel_preference = _select_hotel_preference(details["style"])
    hotel_budget = _estimate_hotel_budget(
        details.get("budget"),
        details["days"],
        details["style"],
    )

    steps: list[dict[str, str]] = []

    flight_input = (
        f"source:{details['source']} destination:{details['destination']} "
        f"preference:{flight_preference}"
    )
    flight_output = search_flights.invoke(flight_input)
    selected_flights = find_flights(
        details["source"],
        details["destination"],
        preference=flight_preference,
    )
    selected_flight = selected_flights[0] if selected_flights else None
    steps.append(_step("search_flights", flight_input, flight_output))

    hotel_parts = [f"city:{details['destination']}"]
    if hotel_budget is not None:
        hotel_parts.append(f"budget:{hotel_budget}")
    hotel_parts.append(f"preference:{hotel_preference}")
    hotel_input = " ".join(hotel_parts)
    hotel_output = recommend_hotels.invoke(hotel_input)
    selected_hotels = find_hotels(
        details["destination"],
        budget=hotel_budget,
        preference=hotel_preference,
    )
    selected_hotel = selected_hotels[0] if selected_hotels else None
    steps.append(_step("recommend_hotels", hotel_input, hotel_output))

    places_input = (
        f"city:{details['destination']} type:{place_type} days:{details['days']}"
    )
    places_output = discover_places.invoke(places_input)
    places_by_day = select_places(
        details["destination"],
        place_type=place_type,
        days=details["days"],
    )
    steps.append(_step("discover_places", places_input, places_output))

    weather_input = f"city:{details['destination']} days:{details['days']}"
    weather_output = get_weather.invoke(weather_input)
    steps.append(_step("get_weather", weather_input, weather_output))

    budget_input = (
        f"flight:{selected_flight['price'] if selected_flight else 0} "
        f"hotel:{selected_hotel['price_per_night'] if selected_hotel else 0} "
        f"days:{details['days']} travelers:{details['travelers']} style:{details['style']}"
    )
    budget_output = estimate_budget.invoke(budget_input)
    budget_data = calculate_budget_breakdown(
        flight_cost=float(selected_flight["price"] if selected_flight else 0),
        hotel_cost=float(selected_hotel["price_per_night"] if selected_hotel else 0),
        days=details["days"],
        travelers=details["travelers"],
        style=details["style"],
    )
    steps.append(_step("estimate_budget", budget_input, budget_output))

    fallback_output = _format_default_itinerary(
        details=details,
        date_range=date_range,
        selected_flight=selected_flight,
        selected_hotel=selected_hotel,
        places_by_day=places_by_day,
        weather_text=weather_output,
        budget_text=budget_output,
    )

    ai_output = _generate_ai_itinerary(
        query=query,
        api_key=(api_key or "").strip(),
        details={**details, "date_range": date_range, "budget_preview": budget_data},
        steps=steps,
    )

    return {
        "output": _sanitize_itinerary_output(ai_output or fallback_output),
        "steps": steps,
        "mode": "ai" if ai_output else "demo",
    }
