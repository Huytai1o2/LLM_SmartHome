# Vector Store & Knowledge Base

This document explains how the RAG (Retrieval-Augmented Generation) pipeline is structured, how the vector store connects to the knowledge base, and how to plug in your own data.

---

## Overview

```
knowledge_base/
├── sources.py          ← defines what gets indexed (edit this)
└── files/              ← place your own documents here (create this folder)

faiss_index/            ← auto-generated FAISS index (do not edit, add to .gitignore)

app/vectore_store/
├── embeddings.py       ← shared embedding model (HuggingFace sentence-transformers)
├── builder.py          ← splits, embeds, and saves documents to faiss_index/
├── loader.py           ← loads a saved index from faiss_index/
└── store.py            ← singleton entry point used by retriever tools
```

---

## How the pipeline works

```
knowledge_base/sources.py
        │
        │  load_documents()  →  List[Document]
        ▼
app/vectore_store/builder.py
        │
        │  split → embed → FAISS.from_documents()
        ▼
faiss_index/                 (persisted to disk)
        │
        ▼
app/vectore_store/loader.py  (loads from disk on next startup)
        │
        ▼
app/vectore_store/store.py   get_vector_store()  ← used by retriever tools
```

1. `sources.py` collects all raw documents from whichever sources you configure.
2. `builder.py` chunks, deduplicates, embeds, and saves them as a FAISS index.
3. On startup, `store.py` loads the saved index (or triggers a rebuild if none exists).
4. Retriever tools call `get_vector_store().similarity_search(query, k=n)`.

---

## Replacing the default dataset with your own data

By default, `knowledge_base/sources.py` loads the public `m-ric/huggingface_doc` dataset from HuggingFace as a placeholder. **You should replace this with your own documents.**

### Step 1 — Create your files folder

Create a `files/` folder inside `knowledge_base/` and place your documents there:

```
knowledge_base/
├── sources.py
└── files/
    ├── company_handbook.pdf
    ├── product_faq.md
    └── internal_notes.txt
```

> **Important:** `knowledge_base/files/` contains confidential data.
> Add it to `.gitignore` so it is never committed to version control:
>
> ```
> knowledge_base/files/
> faiss_index/
> ```

### Step 2 — Update `sources.py`

Open `knowledge_base/sources.py` and edit the `load_documents()` function.
Remove (or comment out) the HuggingFace dataset source and uncomment the local file loaders:

```python
def load_documents() -> List[Document]:
    docs: List[Document] = []

    # --- Your own local files ---
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    loaders = [
        DirectoryLoader("knowledge_base/files/", glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader("knowledge_base/files/", glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader("knowledge_base/files/", glob="**/*.md",  loader_cls=UnstructuredMarkdownLoader),
    ]
    for loader in loaders:
        docs.extend(loader.load())

    return docs
```

### Step 3 — Rebuild the index

Delete the existing `faiss_index/` folder and trigger a rebuild:

```bash
rm -rf faiss_index/
python -c "from app.vectore_store.builder import build_and_save; build_and_save()"
```

The new index will reflect your documents and be saved back to `faiss_index/`.

---

## Supported file formats

| Format  | Loader class                 |
| ------- | ---------------------------- |
| `.txt`  | `TextLoader`                 |
| `.pdf`  | `PyPDFLoader`                |
| `.md`   | `UnstructuredMarkdownLoader` |
| `.docx` | `Docx2txtLoader`             |
| `.csv`  | `CSVLoader`                  |

Add more loaders as needed — the full list is available in the [LangChain document loaders docs](https://python.langchain.com/docs/integrations/document_loaders/).

---

## Environment variables

| Variable               | Default              | Description                           |
| ---------------------- | -------------------- | ------------------------------------- |
| `FAISS_INDEX_PATH`     | `faiss_index`        | Where the FAISS index is saved/loaded |
| `EMBEDDING_MODEL_NAME` | `thenlper/gte-small` | HuggingFace model used for embeddings |
| `CHUNK_SIZE`           | `200`                | Token chunk size for text splitting   |
| `CHUNK_OVERLAP`        | `20`                 | Overlap between consecutive chunks    |
