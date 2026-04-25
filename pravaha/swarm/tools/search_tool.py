"""Search Tool — DuckDuckGo web search (no API key needed)."""

from __future__ import annotations

import logging
import urllib.parse

logger = logging.getLogger(__name__)


class SearchTool:
    """Search the web via DuckDuckGo Instant Answer API (no API key)."""

    name = "web_search"
    description = "Search the web and return top results"
    arg_schema = '{"query": "string", "max_results": 5}'

    def execute(self, query: str, max_results: int = 5) -> dict:
        """Search the web and return results."""
        max_results = min(max_results, 10)
        try:
            import httpx

            encoded = urllib.parse.quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
            resp = httpx.get(url, timeout=10, follow_redirects=True)
            data = resp.json()

            results: list[dict] = []

            # Abstract (direct answer)
            if data.get("Abstract"):
                results.append({
                    "text": data["Abstract"][:200],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for r in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(r, dict) and "Text" in r:
                    results.append({
                        "text": r["Text"][:200],
                        "url": r.get("FirstURL", ""),
                    })
                elif isinstance(r, dict) and "Topics" in r:
                    # Sub-topics
                    for sub in r["Topics"][:2]:
                        if "Text" in sub:
                            results.append({
                                "text": sub["Text"][:200],
                                "url": sub.get("FirstURL", ""),
                            })

            return {
                "query": query,
                "results": results[:max_results],
                "success": True,
            }
        except Exception as e:
            return {"error": str(e), "query": query, "success": False}
