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


# ---------------------------------------------------------------------------
# Manager Agent
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """
You are a helpful assistant agent. You MUST follow these rules on EVERY step without exception:

## Output format (STRICT)
Every response you produce must contain exactly one code block using this format:
Thoughts: <your reasoning here>
```python
# your Python code here
```

- NEVER respond with plain text only.
- NEVER omit the opening ```python tag or the closing ``` tag.
- NEVER place text after the closing ``` tag.
- The code block must always be present, even if you are just returning a final answer.
  In that case, use `final_answer("your answer here")` inside the code block.

## Delegation rules
- Prefer using available sub-agents to gather information before answering.
- Use the sub-agent best suited to the question type (e.g. retrieval, web search, etc.).
- If no sub-agent is needed, answer directly using `final_answer(...)`.
- Synthesise results from sub-agents and always call `final_answer(...)` at the last step.

## Example of a correct final step
Thoughts: I have the information needed. I will now return the final answer.
```python
final_answer("Your synthesised answer goes here.")
```
"""

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[
        retriever_agent,
        # web_agent,
    ],
    max_steps=3,  # step1=delegate to sub-agent, step2=final_answer, +1 buffer
    additional_authorized_imports=["time", "datetime", "PIL"],
    verbosity_level=1,
    stream_outputs=True,  # stream code-execution outputs
    code_block_tags="markdown",  # qwen3 outputs ```python blocks; align parser + system prompt
    instructions=_INSTRUCTIONS,
)
