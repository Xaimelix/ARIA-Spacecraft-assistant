import os
import pickle
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def _get_embeddings_model():
    """Return an embeddings implementation.

    Priority:
    1) If EMBEDDINGS_PROVIDER=openai and OPENAI_API_KEY set -> OpenAIEmbeddings
    2) Otherwise -> HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER", "hf").lower()
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    # Default to HF to avoid external paid APIs by default
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest(
    pdf_path: str = "ИИ в космосе.pdf",
    output_dir: str = "rag_store",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    os.makedirs(output_dir, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    # Add metadata (example values from the request)
    today_str = datetime.now().strftime("%Y-%m-%d")
    for doc in chunks:
        doc.metadata.update(
            {
                "system": "life_support",
                "priority": "critical",
                "last_updated": os.getenv("DOC_LAST_UPDATED", today_str),
                "source": os.path.basename(pdf_path),
            }
        )

    embeddings = _get_embeddings_model()

    # Build and persist FAISS vector index
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    faiss_path = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(faiss_path)

    # Persist chunks for later BM25 reconstruction and debugging
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    return {
        "chunks_count": len(chunks),
        "faiss_path": faiss_path,
        "chunks_path": os.path.join(output_dir, "chunks.pkl"),
    }


if __name__ == "__main__":
    info = ingest()
    print(
        f"Ingestion complete. Chunks: {info['chunks_count']}. FAISS: {info['faiss_path']}. Docs: {info['chunks_path']}"
    )


