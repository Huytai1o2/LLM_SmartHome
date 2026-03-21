"""
Orchestrator – Manager Agent

Implements the top-level CodeAgent that:
  1. Receives the user query.
  2. Delegates to one or more managed sub-agents as needed:
       - web_agent       : live web search & page visit
       - retriever_agent : FAISS-backed HF docs & PEFT issues retrieval
       - image_agent     : prompt refinement + image generation
  3. Has direct access to a Python code interpreter tool.
  4. Synthesises all results into a final answer via the LLM.

Model: Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API (no OpenAI).

Note: In smolagents >=1.0, sub-agents are passed directly to managed_agents as
      MultiStepAgent instances. Each sub-agent must have a name and description
      set so the manager knows when to delegate to them.
"""

from smolagents import CodeAgent

from app.multiple_agentic_system.model import model
from app.multiple_agentic_system.agents.web_agent import web_agent
from app.multiple_agentic_system.agents.retriever_agent import retriever_agent


# ---------------------------------------------------------------------------
# Manager Agent
# ---------------------------------------------------------------------------

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[
        web_agent,
        retriever_agent,
    ],
    additional_authorized_imports=["time", "datetime", "PIL"],
)
