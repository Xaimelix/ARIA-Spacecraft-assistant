import argparse
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader

from vector_base import (
    update_knowledge_base,
    refresh_document,
    load_vectorstore,
)


def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


def cmd_ingest(pdf_path: str, doc_id: str) -> None:
    docs = load_pdf(pdf_path)
    update_knowledge_base(docs, doc_id=doc_id)
    print(f"Ingested '{pdf_path}' as doc_id='{doc_id}'.")


def cmd_refresh(doc_id: str, content: str) -> None:
    refresh_document(doc_id=doc_id, new_content=content)
    print(f"Refreshed doc_id='{doc_id}' with new content.")


def cmd_query(query: str, k: int) -> None:
    store = load_vectorstore()
    if store is None:
        print("Vector store is empty. Ingest a document first.")
        return
    results = store.similarity_search(query, k=k)
    print(f"Top {len(results)} results for query: {query}\n")
    for i, d in enumerate(results, 1):
        meta = d.metadata or {}
        prefix = f"[{meta.get('doc_id', 'unknown')}] " if meta else ""
        print(f"{i}. {prefix}{d.page_content[:300].strip()}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spacecraft RAG Assistant")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into the KB")
    p_ingest.add_argument("--pdf", default="ИИ в космосе.pdf", help="Path to PDF file")
    p_ingest.add_argument(
        "--doc-id",
        default="ИИ в космосе.pdf",
        help="Logical document id to use for updates",
    )

    p_refresh = sub.add_parser("refresh", help="Refresh a logical document by id")
    p_refresh.add_argument("--doc-id", required=True)
    p_refresh.add_argument("--content", required=True, help="New document content")

    p_query = sub.add_parser("query", help="Query the KB")
    p_query.add_argument("--q", required=True, help="Question to search/answer")
    p_query.add_argument("--k", type=int, default=4, help="Top-K chunks to return")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd:
        # Быстрый сценарий по умолчанию: заинжестить PDF и выполнить пример запроса
        pdf_default = "ИИ в космосе.pdf"
        if os.path.exists(pdf_default):
            cmd_ingest(pdf_default, doc_id=pdf_default)
            cmd_query("Кратко опиши ключевые темы документа", k=3)
        else:
            print("Файл 'ИИ в космосе.pdf' не найден. Запустите с подкомандой --help.")
    elif args.cmd == "ingest":
        cmd_ingest(args.pdf, args.doc_id)
    elif args.cmd == "refresh":
        cmd_refresh(args.doc_id, args.content)
    elif args.cmd == "query":
        cmd_query(args.q, args.k)
    else:
        parser.print_help()
