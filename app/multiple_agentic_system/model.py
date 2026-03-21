"""
Shared LLM model used by all agents in the system.

Model  : Qwen/Qwen3.5-4B  (via HuggingFace Inference API)
SDK    : smolagents InferenceClientModel  (no OpenAI dependency)

Notes:
  - unsloth/Qwen3.5-4B-GGUF is a local-only GGUF model and cannot be used
    with InferenceClientModel. The canonical HuggingFace model ID is used here.
  - Override the model via LLM_MODEL_ID in .env if needed.

Free-tier alternatives (set LLM_MODEL_ID in .env to override):
  - Qwen/Qwen3.5-4B          (default)
  - Qwen/Qwen2.5-7B-Instruct  (previous default, stable)
  - meta-llama/Llama-3.1-8B-Instruct
"""

from smolagents import OpenAIServerModel

model = OpenAIServerModel(
    model_id="qwen3",  # qwen3 supports tool calling; gemma3 does not
    api_base="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
    api_key="ollama",  # Required by the client but ignored by Ollama
)
