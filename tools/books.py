from __future__ import annotations

from typing import Any, Dict, List

import requests

from mcp_runtime.registry import mcp
from tools.shared import error_payload, get_json


@mcp.tool()
def book_recs(topic: str, limit: int = 5) -> Dict[str, Any]:
    """Simple book suggestions for a topic via Open Library search."""
    safe_limit = max(1, min(limit, 10))

    try:
        data = get_json(
            "https://openlibrary.org/search.json",
            params={"q": topic, "limit": safe_limit},
        )
    except requests.RequestException as exc:
        return error_payload("books", exc)

    docs = data.get("docs", [])
    picks: List[Dict[str, Any]] = []
    for doc in docs[:safe_limit]:
        picks.append(
            {
                "title": doc.get("title"),
                "author": (doc.get("author_name") or ["Unknown"])[0],
                "year": doc.get("first_publish_year"),
                "work": doc.get("key"),
            }
        )

    return {"topic": topic, "count": len(picks), "results": picks}
