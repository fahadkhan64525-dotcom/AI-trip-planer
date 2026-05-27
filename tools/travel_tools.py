"""
tools/travel_tools.py
---------------------
LangChain tools and reusable helpers for the travel planning assistant.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
from langchain.tools import tool

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
FLIGHTS_PATH = DATA_DIR / "flights.json"
HOTELS_PATH = DATA_DIR / "hotels.json"
PLACES_PATH = DATA_DIR / "places.json"

# ─── City → Coordinates Lookup (for weather API) ─────────────────────────────
CITY_COORDS = {
    "goa": (15.2993, 74.1240),
    "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090),
    "kerala": (10.8505, 76.2711),
    "rajasthan": (27.0238, 74.2179),
    "manali": (32.2432, 77.1892),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "chennai": (13.0827, 80.2707),
    "jaipur": (26.9124, 75.7873),
}

FIELD_PATTERN = re.compile(r"(\w+)\s*:\s*(.+?)(?=\s+\w+\s*:|$)")


@lru_cache(maxsize=3)
def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load and cache a JSON dataset."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_tool_query(query: str) -> dict[str, str]:
    """Parse `key:value` tool input while preserving spaces inside values."""
    return {
        match.group(1).strip().lower(): match.group(2).strip()
        for match in FIELD_PATTERN.finditer(query)
    }


def find_flights(
    source: str,
    destination: str,
    preference: str = "cheapest",
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return flight options sorted by the requested preference."""
    flights = _load_json(FLIGHTS_PATH)
    source = source.strip().lower()
    destination = destination.strip().lower()
    preference = preference.strip().lower() or "cheapest"

    matches = [
        flight
        for flight in flights
        if source in flight["source"].lower()
        and destination in flight["destination"].lower()
    ]

    if preference == "fastest":
        matches.sort(key=lambda item: (item["duration_hrs"], item["price"]))
    elif preference == "best":
        matches.sort(key=lambda item: (item["price"], item["duration_hrs"]))
    else:
        matches.sort(key=lambda item: (item["price"], item["duration_hrs"]))

    return matches[:limit]


def find_hotels(
    city: str,
    budget: int | None = None,
    preference: str = "rating",
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Return hotel options, relaxing the budget if it filters everything out."""
    hotels = _load_json(HOTELS_PATH)
    city = city.strip().lower()
    preference = preference.strip().lower() or "rating"

    city_matches = [hotel for hotel in hotels if city in hotel["city"].lower()]
    if not city_matches:
        return []

    if budget is None:
        matches = city_matches
    else:
        matches = [
            hotel for hotel in city_matches
            if hotel["price_per_night"] <= budget
        ] or city_matches

    if preference == "cheapest":
        matches.sort(key=lambda item: (item["price_per_night"], -item["rating"]))
    elif preference == "luxury":
        matches.sort(key=lambda item: (-item["stars"], -item["rating"], item["price_per_night"]))
    else:
        matches.sort(key=lambda item: (-item["rating"], -item["stars"], item["price_per_night"]))

    return matches[:limit]


def select_places(
    city: str,
    place_type: str = "All",
    days: int = 3,
) -> list[list[dict[str, Any]]]:
    """Return day-wise place recommendations."""
    places = _load_json(PLACES_PATH)
    city = city.strip().lower()
    place_type = place_type.strip().title() or "All"
    days = max(int(days), 1)

    matches = [place for place in places if city in place["city"].lower()]
    if not matches:
        return []

    if place_type != "All":
        typed_matches = [
            place for place in matches
            if place["type"].lower() == place_type.lower()
        ]
        if typed_matches:
            matches = typed_matches

    matches.sort(key=lambda item: -item["rating"])
    per_day = 2 if days >= 4 else 3
    selected = matches[: days * per_day]

    return [
        selected[index:index + per_day]
        for index in range(0, len(selected), per_day)
    ]


def _interpret_weather(code: int, rain: float | int | None) -> str:
    """Convert a WMO weather code into a concise label."""
    if code == 0:
        return "☀️ Clear Sky"
    if code in (1, 2):
        return "⛅ Partly Cloudy"
    if code == 3:
        return "☁️ Overcast"
    if code in (51, 53, 55):
        return "🌦️ Light Drizzle"
    if code in (61, 63, 65):
        return "🌧️ Rain"
    if code in (71, 73, 75):
        return "❄️ Snow"
    if code in (80, 81, 82):
        return "🌧️ Rain Showers"
    if code in (95, 96, 99):
        return "⛈️ Thunderstorm"
    if rain and rain > 5:
        return "🌧️ Rainy"
    return "🌤️ Mostly Clear"


def fetch_weather_forecast(city: str, days: int = 3) -> dict[str, Any]:
    """Fetch structured weather data for known destinations."""
    city_name = city.strip().title()
    city_key = city.strip().lower()
    forecast_days = min(max(int(days), 1), 7)
    coords = CITY_COORDS.get(city_key)

    if not coords:
        return {
            "city": city_name,
            "days": forecast_days,
            "forecast": [],
            "error": f"Weather data not available for {city_name}. Showing estimated conditions.",
        }

    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
        f"&timezone=auto&forecast_days={forecast_days}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        daily = response.json().get("daily", {})
    except requests.RequestException as exc:
        return {
            "city": city_name,
            "days": forecast_days,
            "forecast": [],
            "error": (
                "⚠️ Weather API unavailable. Estimated: Warm and pleasant for travel. "
                f"(Error: {exc})"
            ),
        }

    dates = daily.get("time", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precipitation = daily.get("precipitation_sum", [])
    weather_codes = daily.get("weathercode", [])

    forecast = []
    for index in range(min(forecast_days, len(dates))):
        rainfall = precipitation[index] if index < len(precipitation) else 0
        forecast.append({
            "date": dates[index],
            "high": max_temps[index] if index < len(max_temps) else "N/A",
            "low": min_temps[index] if index < len(min_temps) else "N/A",
            "rainfall": rainfall,
            "condition": _interpret_weather(
                weather_codes[index] if index < len(weather_codes) else 0,
                rainfall,
            ),
        })

    return {
        "city": city_name,
        "days": forecast_days,
        "forecast": forecast,
        "error": None,
    }


def calculate_budget_breakdown(
    flight_cost: float,
    hotel_cost: float,
    days: int = 3,
    travelers: int = 1,
    style: str = "moderate",
) -> dict[str, Any]:
    """Return a structured trip budget."""
    days = max(int(days), 1)
    travelers = max(int(travelers), 1)
    style = style.strip().lower() or "moderate"

    daily_expenses = {
        "budget": {"food": 500, "transport": 300, "activities": 200, "misc": 200},
        "moderate": {"food": 1200, "transport": 600, "activities": 500, "misc": 400},
        "luxury": {"food": 3000, "transport": 1500, "activities": 1500, "misc": 1000},
    }
    profile = daily_expenses.get(style, daily_expenses["moderate"])

    flight_total = flight_cost * travelers
    hotel_total = hotel_cost * days
    food_total = profile["food"] * days * travelers
    transport_total = profile["transport"] * days
    activities_total = profile["activities"] * days * travelers
    misc_total = profile["misc"] * days * travelers
    grand_total = (
        flight_total
        + hotel_total
        + food_total
        + transport_total
        + activities_total
        + misc_total
    )

    return {
        "style": style,
        "travelers": travelers,
        "days": days,
        "flight_cost": flight_cost,
        "hotel_cost": hotel_cost,
        "daily_expenses": profile,
        "flight_total": flight_total,
        "hotel_total": hotel_total,
        "food_total": food_total,
        "transport_total": transport_total,
        "activities_total": activities_total,
        "misc_total": misc_total,
        "grand_total": grand_total,
        "per_person": grand_total / travelers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: Flight Search
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_flights(query: str) -> str:
    """
    Search for available flights between two cities.
    Input format: 'source:Delhi destination:Goa preference:cheapest'
    Preferences can be 'cheapest', 'fastest', or 'best'.
    """
    try:
        parts = parse_tool_query(query)
        source = parts.get("source", "").title()
        destination = parts.get("destination", "").title()
        preference = parts.get("preference", "cheapest").lower()
        matches = find_flights(source, destination, preference=preference)

        if not matches:
            return f"No flights found from {source} to {destination}."

        output = f"✈️ Flights from {source} to {destination} ({preference}):\n"
        for flight in matches:
            output += (
                f"  • [{flight['flight_id']}] {flight['airline']} | "
                f"₹{flight['price']:,} | Departs {flight['departure']} → Arrives {flight['arrival']} | "
                f"{flight['duration_hrs']}h | {flight['class']}\n"
            )
        output += (
            f"\n✅ Recommended: {matches[0]['airline']} at ₹{matches[0]['price']:,} "
            f"({matches[0]['departure']} departure)"
        )
        return output
    except Exception as exc:
        return f"Error searching flights: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Hotel Recommendation
# ─────────────────────────────────────────────────────────────────────────────
@tool
def recommend_hotels(query: str) -> str:
    """
    Recommend hotels in a given city.
    Input format: 'city:Goa budget:3000 preference:rating'
    Preference can be 'rating', 'cheapest', or 'luxury'.
    """
    try:
        parts = parse_tool_query(query)
        city = parts.get("city", "").title()
        budget = int(parts["budget"]) if parts.get("budget") else None
        preference = parts.get("preference", "rating").lower()
        matches = find_hotels(city, budget=budget, preference=preference)

        if not matches:
            return f"No hotels found in {city}."

        output = f"🏨 Hotels in {city} (preference: {preference}):\n"
        for hotel in matches:
            amenities = ", ".join(hotel["amenities"][:4])
            output += (
                f"  • [{hotel['hotel_id']}] {hotel['name']} | "
                f"₹{hotel['price_per_night']:,}/night | "
                f"⭐ {hotel['rating']} ({hotel['stars']}★) | {amenities}\n"
            )
        top = matches[0]
        output += (
            f"\n✅ Recommended: {top['name']} at ₹{top['price_per_night']:,}/night "
            f"(Rating: {top['rating']}/5)"
        )
        return output
    except Exception as exc:
        return f"Error recommending hotels: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: Places Discovery
# ─────────────────────────────────────────────────────────────────────────────
@tool
def discover_places(query: str) -> str:
    """
    Discover tourist attractions and points of interest in a city.
    Input format: 'city:Goa type:Beach days:3'
    """
    try:
        parts = parse_tool_query(query)
        city = parts.get("city", "")
        place_type = parts.get("type", "All").title()
        days = int(parts.get("days", 3))
        day_groups = select_places(city, place_type=place_type, days=days)

        if not any(day_groups):
            return f"No places found in {city}."

        output = f"📍 Places to visit in {city} ({days} days):\n"
        for index, day_places in enumerate(day_groups, 1):
            output += f"\n  Day {index}:\n"
            for place in day_places:
                fee = f"₹{place['entry_fee']}" if place["entry_fee"] > 0 else "Free"
                output += (
                    f"    → {place['name']} ({place['type']}) | "
                    f"⭐ {place['rating']} | Entry: {fee} | Best time: {place['best_time']}\n"
                    f"       {place['description']}\n"
                )
        return output
    except Exception as exc:
        return f"Error discovering places: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: Weather Lookup
# ─────────────────────────────────────────────────────────────────────────────
@tool
def get_weather(query: str) -> str:
    """
    Get weather forecast for a destination city.
    Input format: 'city:Goa days:3'
    """
    try:
        parts = parse_tool_query(query)
        city = parts.get("city", "")
        days = int(parts.get("days", 3))
        weather = fetch_weather_forecast(city, days=days)

        if weather["error"]:
            return weather["error"]

        output = f"🌤️ Weather Forecast for {weather['city']} ({weather['days']} days):\n"
        for index, day in enumerate(weather["forecast"], 1):
            output += (
                f"  Day {index} ({day['date']}): {day['condition']} | "
                f"High: {day['high']}°C, Low: {day['low']}°C"
            )
            if day["rainfall"]:
                output += f" | Rainfall: {day['rainfall']}mm"
            output += "\n"

        return output
    except Exception as exc:
        return f"Error fetching weather: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: Budget Estimation
# ─────────────────────────────────────────────────────────────────────────────
@tool
def estimate_budget(query: str) -> str:
    """
    Estimate the total travel budget for a trip.
    Input format: 'flight:4200 hotel:3200 days:3 travelers:2 style:moderate'
    """
    try:
        parts = parse_tool_query(query)
        flight_cost = float(parts.get("flight", 0))
        hotel_cost = float(parts.get("hotel", 0))
        days = int(parts.get("days", 3))
        travelers = int(parts.get("travelers", 1))
        style = parts.get("style", "moderate").lower()
        budget = calculate_budget_breakdown(
            flight_cost=flight_cost,
            hotel_cost=hotel_cost,
            days=days,
            travelers=travelers,
            style=style,
        )
        profile = budget["daily_expenses"]

        return f"""💰 Budget Breakdown ({style.title()} style, {travelers} traveler(s), {days} days):

  ✈️  Flights       : ₹{budget['flight_total']:>8,.0f}   (₹{flight_cost:,.0f} × {travelers} person(s))
  🏨  Hotel         : ₹{budget['hotel_total']:>8,.0f}   (₹{hotel_cost:,.0f}/night × {days} nights)
  🍽️  Food          : ₹{budget['food_total']:>8,.0f}   (₹{profile['food']:,}/person/day)
  🚕  Local Travel  : ₹{budget['transport_total']:>8,.0f}   (₹{profile['transport']:,}/day)
  🎭  Activities    : ₹{budget['activities_total']:>8,.0f}   (₹{profile['activities']:,}/person/day)
  🛍️  Miscellaneous : ₹{budget['misc_total']:>8,.0f}   (₹{profile['misc']:,}/person/day)
  {'─' * 45}
  💵  TOTAL         : ₹{budget['grand_total']:>8,.0f}
  👤  Per Person    : ₹{budget['per_person']:>8,.0f}
"""
    except Exception as exc:
        return f"Error estimating budget: {exc}"


# ─── Export all tools as a list ───────────────────────────────────────────────
ALL_TOOLS = [
    search_flights,
    recommend_hotels,
    discover_places,
    get_weather,
    estimate_budget,
]
