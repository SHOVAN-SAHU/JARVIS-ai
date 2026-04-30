"""
tools/websearch_tool.py
Web search tool using DuckDuckGo (no API key required).
"""
from typing import Type

from duckduckgo_search import DDGS
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the internet for current information, news, facts, or any topic. "
        "Use this when you need up-to-date information. "
        "Input should be a clear search query."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    before_action_phrase: str = "Alright, searching the web for that now."
    max_results: int = 4

    def _run(self, query: str) -> str:
        logger.info(f"🔍  Web search: '{query}'")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return "I couldn't find any results for that query."

            # Format results into readable text
            formatted = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", "")
                formatted.append(f"{i}. {title}: {body}")

            return "\n\n".join(formatted)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Search failed: {e}"

    async def _arun(self, query: str) -> str:
        return self._run(query)