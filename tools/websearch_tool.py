"""
tools/websearch_tool.py
Web search using Tavily via langchain-tavily.
Rotates through up to 3 API keys if one fails or is rate-limited.
"""
import logging
import os
import time
import random
from typing import Type, List

from langchain_tavily import TavilySearch
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the internet for CURRENT or REAL-TIME information only: "
        "live news, today's prices, exchange rates, sports scores, weather, recent events. "
        "Do NOT use this for general knowledge, greetings, opinions, math, definitions, "
        "or anything a knowledgeable person could answer without internet access."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    before_action_phrase: str = "Searching the web for that now."
    max_results: int = 4

    def _get_api_keys(self) -> List[str]:
        """Return all configured Tavily API keys, skipping empty ones."""
        keys = [
            settings.TAVILY_API_KEY_1,
            settings.TAVILY_API_KEY_2,
            settings.TAVILY_API_KEY_3,
        ]
        return [k for k in keys if k.strip()]

    def _try_search(self, query: str, api_key: str, topic: str = "general") -> list:
        """Attempt a single search with a specific API key."""
        os.environ["TAVILY_API_KEY"] = api_key
        tool = TavilySearch(
            max_results=self.max_results,
            topic=topic,
            search_depth="basic",
        )
        response = tool.invoke({"query": query})
        return response.get("results", [])

    def _search_with_key_rotation(self, query: str) -> list:
        """
        Try each API key in order.
        If a key fails, wait briefly and move to the next one.
        Final fallback: retry all keys with a simplified query + news topic.
        """
        api_keys = self._get_api_keys()

        if not api_keys:
            logger.error("No Tavily API keys configured.")
            return []

        # --- Pass 1: try each key with the original query ---
        for i, key in enumerate(api_keys, start=1):
            try:
                logger.info(f"Tavily: trying key {i}/{len(api_keys)}")
                results = self._try_search(query, key, topic="general")
                if results:
                    logger.info(f"Tavily: success with key {i}")
                    return results
                logger.warning(f"Tavily: key {i} returned empty results")
            except Exception as e:
                logger.warning(f"Tavily: key {i} failed — {e}")

            # Small delay before trying the next key
            if i < len(api_keys):
                time.sleep(1.0 + random.uniform(0, 0.5))

        # --- Pass 2: fallback — simplified query + news topic, all keys ---
        simplified = " ".join(query.split()[:5])
        logger.info(f"Tavily: all keys failed on original query, trying fallback '{simplified}'")

        for i, key in enumerate(api_keys, start=1):
            try:
                logger.info(f"Tavily fallback: trying key {i}/{len(api_keys)}")
                results = self._try_search(simplified, key, topic="news")
                if results:
                    logger.info(f"Tavily fallback: success with key {i}")
                    return results
                logger.warning(f"Tavily fallback: key {i} returned empty results")
            except Exception as e:
                logger.warning(f"Tavily fallback: key {i} failed — {e}")

            if i < len(api_keys):
                time.sleep(1.0 + random.uniform(0, 0.5))

        return []

    def _run(self, query: str) -> str:
        logger.info(f"Web search: '{query}'")
        results = self._search_with_key_rotation(query)

        if not results:
            return (
                "Web search is currently unavailable. "
                "Please answer from your own knowledge if possible, "
                "or ask the user to try again in a few seconds."
            )
        
        print(f"websearch results: {results}")

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            formatted.append(f"{i}. {title}: {content}\n   Source: {url}")

        return "\n\n".join(formatted)

    async def _arun(self, query: str) -> str:
        return self._run(query)