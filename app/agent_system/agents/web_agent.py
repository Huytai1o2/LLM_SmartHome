"""
Managed Web Agent

Equipped with:
  - DuckDuckGoSearch tool  : searches the web for up-to-date information.
  - VisitWebpage tool      : fetches and parses the content of a webpage.
"""

from smolagents import ToolCallingAgent
from app.agent_system.model import model
from app.agent_system.tools.web_tools import search_tool

web_agent = ToolCallingAgent(
    tools=[search_tool],  # VisitWebpageTool removed — page fetching adds 2-5s per call
    model=model,
    max_steps=2,  # search once, summarise once
    verbosity_level=1,
    name="search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)
