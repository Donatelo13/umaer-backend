
import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename   # <-- IMPORTANTE

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Límite y extensiones permitidas
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}

def allowed_file(name_or_path) -> bool:
    # Acepta str o Path
    suffix = Path(name_or_path).suffix.lower()
    return suffix in ALLOWED_EXTS

def ensure_session_dir(session_id: str) -> Path:
    ses_dir = UPLOAD_ROOT / session_id
    ses_dir.mkdir(parents=True, exist_ok=True)
    return ses_dir

def save_file(file_storage, dest_dir: Path) -> Path:
    # Nombre seguro
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}")

    # Validar extensión antes de guardar
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        return f"[No se pudo extraer texto de {pdf_path.name}: {e}]"

def collect_session_corpus(ses_dir: Path) -> str:
    corpus = []
    for p in ses_dir.iterdir():
        if p.suffix.lower() == ".pdf":
            corpus.append(f"\n--- {p.name} ---\n")
            corpus.append(extract_text_from_pdf(p))
    return "\n".join(corpus).strip()

@app.get("/api/health")
def health():
    return jsonify(status="ok")

@app.post("/api/chat")
def chat():
    message = (request.form.get("message") or "").strip()
    session_id = request.form.get("session_id") or uuid.uuid4().hex

    if not message and "files" not in request.files:
        return jsonify(error="Envía al menos 'message' o algún archivo."), 400

    ses_dir = ensure_session_dir(session_id)

    saved_files = []
    try:
        if "files" in request.files:
            for f in request.files.getlist("files"):
                if not f or not f.filename:
                    continue
                dest = save_file(f, ses_dir)
                saved_files.append(dest.name)
    except ValueError as ve:
        # Por extensión no permitida, etc.
        return jsonify(error=str(ve)), 415

    corpus = collect_session_corpus(ses_dir)

    if corpus:
        reply = (
            "He revisado los archivos de tu sesión y esto es un ejemplo de respuesta.\n\n"
            f"Tu mensaje: {message or '(vacío)'}\n\n"
            f"Archivos en sesión: {', '.join(saved_files) if saved_files else '(sin nuevos)'}\n\n"
            "Resumen (muy básico) del corpus (primeros 500 chars):\n"
            + corpus[:500]
        )
    else:
        reply = (
            "No he encontrado archivos en tu sesión. "
            f"Tu mensaje fue: {message or '(vacío)'}.\n"
            "Adjunta PDFs (o habilita OCR para imágenes) para que pueda responder según su contenido."
        )

    return jsonify(session_id=session_id, reply=reply, used_files=saved_files)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
