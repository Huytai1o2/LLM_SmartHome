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

from app.agent_system.model import model
from app.agent_system.agents.web_agent import web_agent
from app.agent_system.agents.retriever_agent import retriever_agent
from app.agent_system.agents.smart_home_agent import smart_home_agent
from app.agent_system.agents.clarification_agent import clarification_agent


# ---------------------------------------------------------------------------
# Manager Agent
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """
You are a helpful IoT smart home assistant agent.

=== ROUTING RULES ===

For SENSOR/STATUS QUERIES (current temperature, brightness, lock state, motion, volume, power, CO2, etc.)
AND for CONTROL COMMANDS (turn on/off, set, lock, unlock, pause, play, open, close, adjust):
```python
result = smart_home_agent(task="<user message verbatim>")
final_answer(result)
```

For GENERAL KNOWLEDGE questions (automation rules, how devices work, device specs, capabilities):
```python
result = retriever_agent(task="<user question>")
final_answer(result)
```

For CLARIFICATION (control command with no room named and device is not thermostat or front door lock):
```python
question = clarification_agent(task="User wants to [verb] the [device] but gave no location. Ask which room.")
final_answer(question)
```

Examples:
- "how bright is the living room light?" → smart_home_agent
- "what is the temperature in bedroom?" → smart_home_agent
- "is the front door locked?" → smart_home_agent
- "turn on the bedroom light" → smart_home_agent
- "set thermostat to 22°C" → smart_home_agent
- "turn off the speaker" (no room) → clarification_agent
- "what automation rules are there?" → retriever_agent
- "what did I ask previously?" → retriever_agent

=== FORMAT ===
Always use Python code blocks. End with final_answer(...). Never plain text only.
Sub-agents return plain text strings — never iterate over them.
"""

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[
        retriever_agent,
        smart_home_agent,
        clarification_agent,
        # web_agent,
    ],
    max_steps=5,  # retrieve → control → final_answer, +2 buffer
    additional_authorized_imports=["time", "datetime", "PIL"],
    verbosity_level=1,
    stream_outputs=True,  # stream code-execution outputs
    code_block_tags="markdown",  # qwen3 outputs ```python blocks; align parser + system prompt
    instructions=_INSTRUCTIONS,
)
