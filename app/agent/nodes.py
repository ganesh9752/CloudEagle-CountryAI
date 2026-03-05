import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_llm
from app.agent.state import AgentState
from app.agent.tools import fetch_country_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available fields (used to detect unsupported requests)
# ---------------------------------------------------------------------------

AVAILABLE_FIELDS = {
    "population", "capital", "currency", "area", "languages",
    "region", "flag", "timezones", "borders", "continent",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are an intent extraction assistant for a country information service.

Your job: parse the user's natural language query and return a JSON object with:
  - "countries": list of country names mentioned (use official/common English names)
  - "fields": list of information fields requested

Valid fields:
  population, capital, currency, area, languages, region, flag, timezones, borders, continent

Rules:
  - If no specific fields are mentioned, default to ["capital", "population", "currency"].
  - If the user asks to "compare" or asks about multiple countries, include all of them.
  - For ambiguous names (e.g. "Georgia"), prefer the sovereign country over the US state.
  - Return ONLY valid JSON — no markdown, no extra text.

IMPORTANT — Handling conversation history:
  - You may receive previous Q&A exchanges as context.
  - If the user uses pronouns like "their", "its", "that country", "those countries", or
    says "what about..." / "and the...", resolve the reference using the conversation history.
  - Always output the fully resolved country names, never leave references unresolved.
  - If the user asks for a field not in the valid list above (e.g. GDP, HDI, literacy rate),
    still include the field name they asked for — the system will handle it downstream.

Examples:
  Query : "What is the population of Germany?"
  Output: {"countries": ["Germany"], "fields": ["population"]}

  Query : "What currency does Japan use?"
  Output: {"countries": ["Japan"], "fields": ["currency"]}

  Query : "Tell me about Brazil"
  Output: {"countries": ["Brazil"], "fields": ["capital", "population", "currency"]}

  Query : "Compare the population and area of India and China"
  Output: {"countries": ["India", "China"], "fields": ["population", "area"]}

  (After discussing India and China)
  Query : "What about their GDP per capita?"
  Output: {"countries": ["India", "China"], "fields": ["gdp per capita"]}
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about countries.

IMPORTANT CONTEXT:
  - You answer questions using data from the REST Countries API (restcountries.com).
  - This API provides ONLY these fields: population, capital, currency, area, languages,
    region, flag, timezones, borders, and continent.
  - It does NOT provide: GDP, GDP per capita, HDI, literacy rate, unemployment, inflation,
    life expectancy, or any economic/social indicators.

Rules:
  - Answer ONLY using the provided country data. Never hallucinate facts.
  - Format numbers with commas (e.g. 83,783,942 for population).
  - If the user asked for a field that is NOT available in the REST Countries API,
    explicitly state: "The REST Countries API I use does not provide [field name] data."
    Do NOT say "the data is missing" or "the data doesn't contain it" — be specific about
    the API limitation.
  - If a field IS available but has a null/empty value, say so explicitly.
  - Be concise and conversational — 1–3 sentences is ideal unless a comparison warrants more.
  - If there is a partial error (some countries fetched, some failed), answer what you can and
    mention the failure for the rest.
"""

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

FIELD_EXTRACTORS: dict[str, Any] = {
    "population": lambda d: {"population": d.get("population")},
    "capital":    lambda d: {"capital": (d.get("capital") or [None])[0]},
    "currency":   lambda d: {
        "currencies": {
            code: info.get("name")
            for code, info in (d.get("currencies") or {}).items()
        }
    },
    "area":       lambda d: {"area_km2": d.get("area")},
    "languages":  lambda d: {"languages": list((d.get("languages") or {}).values())},
    "region":     lambda d: {"region": d.get("region"), "subregion": d.get("subregion")},
    "flag":       lambda d: {
        "flag_emoji": d.get("flag"),
        "flag_url": (d.get("flags") or {}).get("png"),
    },
    "timezones":  lambda d: {"timezones": d.get("timezones")},
    "borders":    lambda d: {"borders": d.get("borders")},
    "continent":  lambda d: {"continents": d.get("continents")},
}


def _extract_fields(country_data: dict, fields: list[str]) -> dict:
    """Return a slim dict with only the requested fields plus the country name."""
    result: dict = {"name": (country_data.get("name") or {}).get("common", "Unknown")}
    for field in fields:
        extractor = FIELD_EXTRACTORS.get(field)
        if extractor:
            result.update(extractor(country_data))
    return result


def _parse_llm_json(content: str) -> dict:
    """Strip markdown fences and parse JSON from LLM output."""
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        # Remove closing fence
        if "```" in content:
            content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def _build_history_context(history: list[dict]) -> str:
    """Format conversation history for inclusion in prompts."""
    if not history:
        return ""
    lines = ["\nConversation history (most recent last):"]
    for entry in history[-5:]:  # last 5 exchanges max
        lines.append(f"  User: {entry.get('question', '')}")
        lines.append(f"  Assistant: {entry.get('answer', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

async def intent_node(state: AgentState) -> dict:
    """LLM step: extract country names and requested fields from the user query."""
    llm = get_llm()

    history_context = _build_history_context(state.get("history", []))
    user_content = state["query"]
    if history_context:
        user_content = f"{history_context}\n\nCurrent question: {user_content}"

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        parsed = _parse_llm_json(response.content)

        countries: list[str] = parsed.get("countries", [])
        fields: list[str] = parsed.get("fields", [])

        if not countries:
            return {
                "countries": [],
                "fields": [],
                "error": "I couldn't identify a country in your question. Please mention a specific country.",
            }

        # Identify which fields are unsupported by the API
        unsupported = [f for f in fields if f not in AVAILABLE_FIELDS]
        supported = [f for f in fields if f in AVAILABLE_FIELDS]

        logger.info(
            "Intent extracted — countries=%s fields=%s unsupported=%s",
            countries, supported, unsupported,
        )
        return {
            "countries": countries,
            "fields": fields,  # pass all fields (including unsupported) for synthesis awareness
            "error": None,
        }

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Intent parsing failed: %s", exc)
        return {
            "countries": [],
            "fields": [],
            "error": "I had trouble understanding that question. Could you rephrase it?",
        }
    except Exception as exc:
        logger.exception("Unexpected error in intent_node")
        return {"countries": [], "fields": [], "error": f"LLM error: {exc}"}


async def tool_node(state: AgentState) -> dict:
    """Tool step: fetch raw country data for every identified country."""
    raw_data: list[dict] = []
    errors: list[str] = []

    for country in state["countries"]:
        result = await fetch_country_data(country)
        if "error" in result:
            errors.append(result["error"])
            logger.warning("API fetch failed for '%s': %s", country, result["error"])
        else:
            raw_data.append(result["data"])
            logger.info("Fetched data for '%s'", country)

    # Hard failure: nothing came back at all
    if not raw_data and errors:
        return {"raw_data": [], "error": " ".join(errors)}

    # Partial failure: some countries worked, some didn't
    partial_error = " ".join(errors) if errors else None
    return {"raw_data": raw_data, "error": partial_error}


async def synthesis_node(state: AgentState) -> dict:
    """LLM step: compose a grounded natural-language answer from the raw data."""
    llm = get_llm()

    # Only extract fields that the API actually provides
    supported_fields = [f for f in state["fields"] if f in AVAILABLE_FIELDS]
    unsupported_fields = [f for f in state["fields"] if f not in AVAILABLE_FIELDS]

    filtered = [_extract_fields(d, supported_fields) for d in state["raw_data"]]
    context = json.dumps(filtered, indent=2, ensure_ascii=False)

    note = f"\nNote: {state['error']}" if state.get("error") else ""

    # Tell the LLM explicitly about unsupported fields
    unsupported_note = ""
    if unsupported_fields:
        unsupported_note = (
            f"\n\nIMPORTANT: The user asked about the following fields which are NOT available "
            f"in the REST Countries API: {', '.join(unsupported_fields)}. "
            f"You must explicitly tell the user that the REST Countries API does not provide "
            f"this data."
        )

    user_prompt = (
        f"User question: {state['query']}\n\n"
        f"Country data:\n{context}"
        f"{note}"
        f"{unsupported_note}\n\n"
        "Please answer the user's question based solely on the provided data."
    )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        return {"answer": response.content}
    except Exception as exc:
        logger.exception("Unexpected error in synthesis_node")
        return {"answer": f"I found the data but encountered an error composing the answer: {exc}"}


async def error_node(state: AgentState) -> dict:
    """Terminal node: return a clean error message when no data could be fetched."""
    error_detail = state.get("error", "An unknown error occurred.")
    return {"answer": f"Sorry, I couldn't answer that question. {error_detail}"}
