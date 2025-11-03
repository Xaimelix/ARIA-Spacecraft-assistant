## Spacecraft Assistant — RAG with YandexGPT

This project implements a basic RAG pipeline using a hybrid retriever (FAISS vector search + BM25 keyword search) and YandexGPT for answer generation.

### Setup

1) Python 3.10+ is recommended. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Place your source documents. By default, the project ingests `ИИ в космосе.pdf` in the project root.

4) Set environment variables for YandexGPT:

```bash
$env:YANDEX_API_KEY="<your-api-key>"
$env:YANDEX_FOLDER_ID="<your-folder-id>"
```

Optional: Choose embeddings provider (default is HuggingFace). To use OpenAI embeddings instead:

```bash
$env:EMBEDDINGS_PROVIDER="openai"
$env:OPENAI_API_KEY="<your-openai-key>"
```

### Ingest

Build the FAISS index and persist chunked docs with metadata:

```bash
python ingest.py
```

Artifacts will be saved under `rag_store/`.

### Ask Questions (RAG)

```bash
python rag_yandex.py "Какова процедура аварийного отключения системы?" --k 5 --filter "system=life_support,priority=critical"
```

Notes:
- Retrieval uses an ensemble of FAISS (70%) and BM25 (30%).
- Metadata filtering is applied to the vector retriever; BM25 searches full corpus.
- Answer generation calls YandexGPT via the Foundation Models API (`completion` endpoint).

### Files

- `ingest.py`: Loads PDF, chunks text, adds metadata, builds FAISS, saves chunks.
- `rag_yandex.py`: Hybrid retrieval + YandexGPT completion, simple CLI.
- `requirements.txt`: Dependencies.
- `vector_base.py`: Legacy file; new pipeline does not depend on it.

### YandexGPT API

The integration uses Yandex Cloud Foundation Models Completion API with `modelUri`:

```
gpt://<FOLDER_ID>/yandexgpt/latest
```

Headers:

```
Authorization: Api-Key <YANDEX_API_KEY>
x-folder-id: <YANDEX_FOLDER_ID>
```

Refer to Yandex Cloud AI Studio docs: https://yandex.cloud/en/docs/ai-studio/concepts/api


