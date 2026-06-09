# A very simple ReAct agent (Reason + Act) that fetches live weather data.
import json
import urllib.parse
import urllib.request

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()  # expects OPENAI_API_KEY in .env


def _get(base, **params):
    url = base + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Input is a plain city name, e.g. "Tokyo"."""
    geo = _get("https://geocoding-api.open-meteo.com/v1/search", name=city, count=1)
    if not geo.get("results"):
        return f"Could not find a location named '{city}'."

    place = geo["results"][0]
    w = _get(
        "https://api.open-meteo.com/v1/forecast",
        latitude=place["latitude"],
        longitude=place["longitude"],
        current_weather="true",
    )["current_weather"]
    return (
        f"Weather in {place['name']}, {place.get('country', '')}: "
        f"{w['temperature']}°C, wind {w['windspeed']} km/h."
    )

agent = create_agent(model=ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[get_weather])

if __name__ == "__main__":
    result = agent.invoke({"messages": [HumanMessage(content=input("target weather input: "))]})
    for msg in result["messages"]:
        msg.pretty_print()