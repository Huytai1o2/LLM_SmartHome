"""
Managed Web Agent

Equipped with:
  - DuckDuckGoSearch tool  : searches the web for up-to-date information.
  - VisitWebpage tool      : fetches and parses the content of a webpage.
"""

from smolagents import ToolCallingAgent
from app.multiple_agentic_system.model import model
from app.multiple_agentic_system.tools.web_tools import search_tool, visit_webpage_tool

web_agent = ToolCallingAgent(
    tools=[search_tool, visit_webpage_tool],
    model=model,
    name="search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)
