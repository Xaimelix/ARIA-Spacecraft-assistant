import argparse
import json
import os
import pickle
from typing import Dict, List

import requests
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


def _get_embeddings_model():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "hf").lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _load_vectorstore(store_dir: str):
    embeddings = _get_embeddings_model()
    return FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _load_chunks(chunks_path: str) -> List[Document]:
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


def _build_hybrid_retriever(
    vectorstore: FAISS,
    chunks: List[Document],
    k: int = 5,
    metadata_filter: Dict[str, str] | None = None,
):
    vector_kwargs = {"k": k}
    if metadata_filter:
        vector_kwargs["filter"] = metadata_filter
    vector_retriever = vectorstore.as_retriever(search_kwargs=vector_kwargs)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )


def _yandex_gpt_complete(messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        raise RuntimeError("Set YANDEX_API_KEY and YANDEX_FOLDER_ID environment variables")

    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }
    payload = {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": max_tokens,
        },
        "messages": messages,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Expected response contains alternatives -> message -> text
    alternatives = data.get("result", {}).get("alternatives", [])
    if not alternatives:
        # Fallback: try direct text field if schema changes
        return data.get("result", {}).get("text", "")
    return alternatives[0].get("message", {}).get("text", "")


def answer_question(
    question: str,
    store_dir: str = os.path.join("rag_store", "faiss_index"),
    chunks_path: str = os.path.join("rag_store", "chunks.pkl"),
    k: int = 5,
    metadata_filter: Dict[str, str] | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
):
    vectorstore = _load_vectorstore(store_dir)
    chunks = _load_chunks(chunks_path)
    retriever = _build_hybrid_retriever(vectorstore, chunks, k=k, metadata_filter=metadata_filter)
    context_docs = retriever.get_relevant_documents(question)

    context_blocks = []
    for i, d in enumerate(context_docs, 1):
        meta = d.metadata or {}
        meta_str = ", ".join([f"{k}={v}" for k, v in meta.items()])
        context_blocks.append(f"[Doc {i}] (metadata: {meta_str})\n{d.page_content}")
    context_text = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "text": (
                "You are a spacecraft assistant. Answer strictly based on CONTEXT. "
                "Cite doc numbers like [Doc N] when relevant. If unsure, say you don't know."
            ),
        },
        {
            "role": "user",
            "text": (
                f"QUESTION: {question}\n\nCONTEXT:\n{context_text}"
            ),
        },
    ]

    answer = _yandex_gpt_complete(messages, temperature=temperature, max_tokens=max_tokens)
    return answer, context_docs


def _parse_filter_arg(filter_str: str | None) -> Dict[str, str] | None:
    if not filter_str:
        return None
    result: Dict[str, str] = {}
    for part in filter_str.split(","):
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result or None


def main():
    parser = argparse.ArgumentParser(description="RAG QA with YandexGPT (hybrid retrieval)")
    parser.add_argument("question", type=str, help="User question")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Metadata filter in key=value pairs separated by commas, e.g. system=life_support,priority=critical",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    metadata_filter = _parse_filter_arg(args.filter)
    answer, _docs = answer_question(
        question=args.question,
        k=args.k,
        metadata_filter=metadata_filter,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("\n== Answer ==\n")
    print(answer)


if __name__ == "__main__":
    main()


