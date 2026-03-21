"""
app.multipleAgenticSystem – multi-agent package

Architecture (smolagents / Qwen2.5-72B, no OpenAI):

    manager_agent (CodeAgent)  ← orchestrator.py
    ├── managed_web_agent       ← web_agent.py
    │   ├── DuckDuckGoSearchTool
    │   └── VisitWebpageTool
    ├── managed_retriever_agent ← rag_agent.py
    │   ├── RetrieverToolHFDocs     (FAISS)
    │   └── RetrieverToolPEFTIssues (FAISS)
    ├── managed_image_agent     ← image_agent.py
    │   ├── PromptGeneratorTool
    │   └── ImageGenerationTool
    └── PythonInterpreterTool   (direct tool on manager)

Public API:
    from app.multipleAgenticSystem.runner import stream_response
"""
