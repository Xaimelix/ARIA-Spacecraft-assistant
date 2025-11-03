import os
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from vector_base import (
    update_knowledge_base,
    create_document,
    load_vectorstore,
)


UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/upload")
    def upload_page():
        return render_template("upload.html")

    @app.post("/api/upload")
    def api_upload():
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "Нет файла в запросе"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"ok": False, "error": "Имя файла пустое"}), 400
        if not allowed_file(file.filename):
            return jsonify({"ok": False, "error": "Разрешены только .pdf и .txt"}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Ingest into vector store
        ext = filename.rsplit(".", 1)[1].lower()
        logical_doc_id = request.form.get("doc_id") or filename

        try:
            documents: list[Document] = []
            if ext == "pdf":
                # Lazy import to keep startup fast
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(save_path)
                documents = loader.load()
            elif ext == "txt":
                text = open(save_path, "r", encoding="utf-8", errors="ignore").read()
                documents = [create_document(text)]

            update_knowledge_base(documents, doc_id=logical_doc_id)
            return jsonify({"ok": True, "doc_id": logical_doc_id, "filename": filename})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/chat")
    def chat_page():
        return render_template("chat.html")

    @app.post("/api/chat")
    def api_chat():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or "").strip()
        if not question:
            return jsonify({"ok": False, "error": "Пустой запрос"}), 400

        store = load_vectorstore()
        if store is None:
            return jsonify({
                "ok": False,
                "error": "База знаний пуста. Загрузите документы на странице загрузки.",
            }), 400

        # Retrieve context
        retriever = store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        print(docs)
        context = "\n\n".join([d.page_content for d in docs])

        # Use OpenAI if configured; otherwise, return heuristic answer
        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        if api_key_present:
            try:
                llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
                messages = [
                    SystemMessage(
                        content=(
                            "Вы — помощник на борту космического корабля. "
                            "Отвечайте кратко и по делу, используя только контекст. "
                            "Если информации недостаточно — честно укажите это.\n\n"
                            f"Контекст:\n{context}"
                        )
                    ),
                    HumanMessage(content=question),
                ]
                answer = llm.invoke(messages).content
            except Exception as e:
                answer = (
                    "LLM недоступен (" + str(e) + "). "
                    "Ниже приведены наиболее релевантные фрагменты из базы знаний.\n\n" + context[:1500]
                )
        else:
            answer = (
                "LLM не настроен (нет переменной OPENAI_API_KEY). "
                "Вот релевантный контекст из базы знаний:\n\n" + context[:1500]
            )

        snippets = [
            {
                "doc_id": (d.metadata or {}).get("doc_id"),
                "preview": d.page_content[:250],
            }
            for d in docs
        ]

        return jsonify({"ok": True, "answer": answer, "snippets": snippets})

    # Static convenience for uploaded files (debug)
    @app.get("/uploads/<path:filename>")
    def uploaded(filename: str):
        return send_from_directory(UPLOAD_DIR, filename)

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=True)


