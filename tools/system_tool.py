"""
tools/system_tool.py
System information tool — CPU, RAM, battery, disk usage.
"""
from typing import Type

import psutil
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class SystemInfoInput(BaseModel):
    query: str = Field(
        default="all",
        description="What to check: 'cpu', 'memory', 'battery', 'disk', or 'all'",
    )


class SystemInfoTool(BaseTool):
    name: str = "get_system_info"
    description: str = (
        "Get system information like CPU usage, RAM usage, battery level, and disk space. "
        "Input can be 'cpu', 'memory', 'battery', 'disk', or 'all' for everything."
    )
    args_schema: Type[BaseModel] = SystemInfoInput
    before_action_phrase: str = "Let me check your system status."

    def _run(self, query: str = "all") -> str:
        q = query.lower().strip()
        results = []

        if q in ("cpu", "all"):
            cpu = psutil.cpu_percent(interval=1)
            cores = psutil.cpu_count()
            results.append(f"CPU Usage: {cpu}% across {cores} cores")

        if q in ("memory", "ram", "all"):
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            results.append(
                f"RAM: {used_gb:.1f} GB used out of {total_gb:.1f} GB ({mem.percent}% used)"
            )

        if q in ("battery", "all"):
            battery = psutil.sensors_battery()
            if battery:
                status = "charging" if battery.power_plugged else "on battery"
                results.append(
                    f"Battery: {battery.percent:.0f}% — {status}"
                )
            else:
                results.append("Battery: No battery detected (desktop system)")

        if q in ("disk", "all"):
            disk = psutil.disk_usage("/")
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            results.append(
                f"Disk: {free_gb:.1f} GB free out of {total_gb:.1f} GB"
            )

        return "\n".join(results) if results else "Could not retrieve system info."

    async def _arun(self, query: str = "all") -> str:
        return self._run(query)