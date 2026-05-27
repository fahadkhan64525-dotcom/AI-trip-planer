"""
utils/helpers.py
----------------
Utility helpers for parsing trip requests and formatting UI output.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any


# ─── Known Locations ──────────────────────────────────────────────────────────
KNOWN_DESTINATIONS = (
    "rajasthan",
    "bangalore",
    "hyderabad",
    "chennai",
    "manali",
    "mumbai",
    "kerala",
    "jaipur",
    "delhi",
    "goa",
)

KNOWN_SOURCES = (
    "bangalore",
    "hyderabad",
    "chennai",
    "mumbai",
    "delhi",
)


def _build_location_pattern(locations: tuple[str, ...]) -> str:
    """Create a longest-first regex alternation for known locations."""
    return "|".join(
        sorted((re.escape(location) for location in locations), key=len, reverse=True)
    )


DESTINATION_PATTERN = _build_location_pattern(KNOWN_DESTINATIONS)
SOURCE_PATTERN = _build_location_pattern(KNOWN_SOURCES)


def _extract_location_after_prefix(
    query_lower: str,
    prefixes: tuple[str, ...],
    pattern: str,
) -> str | None:
    """Extract a known location that appears after a source/destination prefix."""
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    match = re.search(
        rf"(?:{prefix_pattern})\s*[:\-]?\s*({pattern})\b",
        query_lower,
    )
    if match:
        return match.group(1).title()
    return None


def _find_first_known_location(
    query_lower: str,
    locations: tuple[str, ...],
    exclude: set[str] | None = None,
) -> str | None:
    """Return the earliest-known location mentioned in the query."""
    exclude = exclude or set()
    matches = []
    for location in locations:
        if location in exclude:
            continue
        index = query_lower.find(location)
        if index != -1:
            matches.append((index, location))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])
    return matches[0][1].title()


def build_trip_date_range(days: int) -> str:
    """Return a near-future date range string for the itinerary."""
    days = max(int(days), 1)
    start_date = datetime.now() + timedelta(days=7)
    end_date = start_date + timedelta(days=days - 1)
    return f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"


def extract_trip_details(user_query: str) -> dict[str, Any]:
    """
    Parse natural-language travel input into structured trip details.

    Returns destination, source, days, budget, travelers, and style.
    """
    query_lower = user_query.lower()
    details: dict[str, Any] = {
        "destination": None,
        "source": "Delhi",
        "days": 3,
        "budget": None,
        "travelers": 1,
        "style": "moderate",
    }

    source = _extract_location_after_prefix(
        query_lower,
        ("from", "source", "source city", "departure city", "departing from"),
        SOURCE_PATTERN,
    )
    if source:
        details["source"] = source

    destination = _extract_location_after_prefix(
        query_lower,
        ("to", "destination", "destination city", "visit"),
        DESTINATION_PATTERN,
    )
    if not destination:
        destination = _find_first_known_location(
            query_lower,
            KNOWN_DESTINATIONS,
            exclude={details["source"].lower()},
        )
    details["destination"] = destination

    day_patterns = (
        r"\b(\d+)\s*[- ]?\s*days?\b",
        r"\b(\d+)\s*[- ]?\s*nights?\b",
        r"\bduration\s*[:\-]?\s*(\d+)\b",
    )
    for pattern in day_patterns:
        match = re.search(pattern, query_lower)
        if match:
            details["days"] = int(match.group(1))
            break

    budget_match = re.search(
        r"(?:budget|₹|rs\.?|inr)\s*(?:of)?\s*[:\-]?\s*(\d[\d,]*)",
        query_lower,
    )
    if budget_match:
        details["budget"] = int(budget_match.group(1).replace(",", ""))

    traveler_patterns = (
        r"\b(\d+)\s*(?:person|people|travelers?|adults?|pax)\b",
        r"(?:for|with)\s+(\d+)\s*(?:of us|people|travelers?)\b",
    )
    for pattern in traveler_patterns:
        match = re.search(pattern, query_lower)
        if match:
            details["travelers"] = int(match.group(1))
            break

    if any(keyword in query_lower for keyword in ("luxury", "premium", "5 star", "five star")):
        details["style"] = "luxury"
    elif any(keyword in query_lower for keyword in ("budget", "cheap", "affordable", "backpack")):
        details["style"] = "budget"

    return details


def build_agent_query(details: dict[str, Any]) -> str:
    """Build a normalized trip-planning query for the planner."""
    destination = details.get("destination") or "Goa"
    source = details.get("source") or "Delhi"
    days = max(int(details.get("days", 3) or 3), 1)
    travelers = max(int(details.get("travelers", 1) or 1), 1)
    style = (details.get("style") or "moderate").lower()
    budget = details.get("budget")
    date_range = build_trip_date_range(days)

    query = (
        f"Plan a {days}-day {style} trip to {destination} for {travelers} travelers. "
        f"Departure city: {source}. "
        f"Trip dates: {date_range}. "
    )

    if budget:
        query += f"Total budget: ₹{int(budget):,}. "

    query += (
        "Please search for the best flights, hotels, places to visit, weather, "
        "and provide a complete itinerary with budget breakdown."
    )

    return query


def format_steps_for_display(steps: list[Any]) -> list[dict[str, str]]:
    """Normalize reasoning steps from either dict-based or LangChain-style traces."""
    formatted = []
    tool_icons = {
        "search_flights": "✈️",
        "recommend_hotels": "🏨",
        "discover_places": "📍",
        "get_weather": "🌤️",
        "estimate_budget": "💰",
    }

    for step in steps:
        if isinstance(step, dict):
            tool_name = str(step.get("tool", "unknown"))
            tool_input = str(step.get("input", ""))
            observation = str(step.get("output", ""))
        else:
            try:
                action, observation = step
            except (TypeError, ValueError):
                action, observation = None, step
            tool_name = str(getattr(action, "tool", "unknown"))
            tool_input = str(getattr(action, "tool_input", ""))
            observation = str(observation)

        formatted.append({
            "tool": tool_name,
            "icon": tool_icons.get(tool_name, "🔧"),
            "input": tool_input,
            "output": observation[:500] + ("..." if len(observation) > 500 else ""),
        })

    return formatted


def validate_api_key(api_key: str) -> bool:
    """Basic validation check for an supported AI service key."""
    if not api_key:
        return False
    api_key = api_key.strip()
    return (
        (api_key.startswith("sk-") and len(api_key) > 20)
        or (api_key.startswith("hf_") and len(api_key) > 20)
    )


def get_destination_tips(destination: str) -> list[str]:
    """Return quick destination-specific travel tips."""
    tips = {
        "Goa": [
            "🌊 Best season: November to February for perfect beach weather",
            "🛵 Rent a scooter — it's the best way to explore the coast",
            "🍹 Try local Goan seafood and feni (cashew liquor) at beach shacks",
        ],
        "Kerala": [
            "🌿 Book your houseboat in advance — they fill up fast",
            "🌧️ Avoid June–August (peak monsoon) unless you love the rains",
            "🐘 Visit during Thrissur Pooram festival for an unforgettable experience",
        ],
        "Rajasthan": [
            "🐪 Best season: October to March — avoid summer heat above 45°C",
            "🏰 Hire a local guide for forts — the stories make it magical",
            "🎨 Shop for handicrafts at Johari Bazaar in Jaipur",
        ],
        "Manali": [
            "❄️ Carry warm layers even in summer — nights can drop to 5°C",
            "🏔️ Acclimatize for a day before attempting Rohtang Pass",
            "🚌 Book buses/cabs to Rohtang Pass early — spots fill fast",
        ],
        "Mumbai": [
            "🚉 Use the local trains — they're the fastest way to get around",
            "🌆 Visit Marine Drive at sunset for the iconic city view",
            "🍱 Try vada pav and pav bhaji from street stalls — it's a Mumbai must",
        ],
    }
    return tips.get(destination, [
        "📱 Download offline maps before your trip",
        "💳 Carry some cash — not all places accept cards",
        "🌐 Check local weather forecasts daily during your trip",
    ])
