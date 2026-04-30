"""
tools/datetime_tool.py
Returns current date and time information.
"""
from datetime import datetime
from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DateTimeInput(BaseModel):
    query: str = Field(default="", description="Optional: 'date', 'time', or 'full'")


class DateTimeTool(BaseTool):
    name: str = "get_current_datetime"
    description: str = (
        "Use this tool to get the current date and/or time. "
        "Input can be 'date', 'time', or leave blank for both."
    )
    args_schema: Type[BaseModel] = DateTimeInput
    before_action_phrase: str = "Checking the current time for you."

    def _run(self, query: str = "") -> str:
        now = datetime.now()
        q = query.lower().strip()

        if q == "date":
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        elif q == "time":
            return f"The current time is {now.strftime('%I:%M %p')}."
        else:
            return (
                f"Today is {now.strftime('%A, %B %d, %Y')} "
                f"and the time is {now.strftime('%I:%M %p')}."
            )

    async def _arun(self, query: str = "") -> str:
        return self._run(query)