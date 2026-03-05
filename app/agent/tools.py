import httpx

COUNTRIES_API_BASE = "https://restcountries.com/v3.1"
REQUEST_TIMEOUT = 10.0


async def fetch_country_data(country_name: str) -> dict:
    """
    Fetch raw country data from REST Countries API.

    Returns either:
      {"data": <country_dict>}   on success (first/best match)
      {"error": <message>}       on failure
    """
    url = f"{COUNTRIES_API_BASE}/name/{country_name}"
    params = {"fullText": "true"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(url, params=params)

            # Fallback: if exact match fails, retry with partial matching
            if response.status_code == 404:
                response = await client.get(url)

            if response.status_code == 404:
                return {"error": f"Country '{country_name}' not found."}

            response.raise_for_status()
            results = response.json()

            # The API returns a list sorted by relevance; the first entry is the best match.
            return {"data": results[0]}

        except httpx.TimeoutException:
            return {"error": f"Request timed out while fetching data for '{country_name}'."}
        except httpx.HTTPStatusError as exc:
            return {"error": f"API error for '{country_name}': HTTP {exc.response.status_code}."}
        except Exception as exc:
            return {"error": f"Unexpected error fetching '{country_name}': {exc}"}
