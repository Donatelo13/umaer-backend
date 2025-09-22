
import os
import uuid
import re
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# === Config Render: usar /tmp ===
UPLOAD_ROOT = Path("/tmp/uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}

# ---------- utilidades de ficheros ----------
def allowed_file(name_or_path) -> bool:
    return Path(name_or_path).suffix.lower() in ALLOWED_EXTS

def ensure_session_dir(session_id: str) -> Path:
    ses_dir = UPLOAD_ROOT / session_id
    ses_dir.mkdir(parents=True, exist_ok=True)
    return ses_dir

def save_file(file_storage, dest_dir: Path) -> Path:
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}")
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido")
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        return f"[Error leyendo {pdf_path.name}: {e}]"

# ---------- utilidades de búsqueda ----------
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")
def tokenize(s: str):
    return [w.lower() for w in _WORD.findall(s)]

def chunk_text(text: str, size=800, overlap=120):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        # extiende al final de frase si hay margen
        if end < n:
            p = text.find(".", end, min(n, end + 200))
            if p != -1:
                chunk = text[start:p+1]
                end = p + 1
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks

def build_index_for_session(session_dir: Path):
    """Devuelve lista de (filename, chunk_text)"""
    index = []
    for p in sorted(session_dir.iterdir()):
        if p.suffix.lower() == ".pdf":
            full = extract_text_from_pdf(p)
            for ch in chunk_text(full):
                index.append((p.name, ch))
    return index

def score_chunk(query_terms, chunk):
    if not chunk:
        return 0.0
    terms = set(tokenize(chunk))
    if not terms:
        return 0.0
    qset = set(query_terms)
    # puntuación = coincidencias únicas / tamaño de query
    return len(qset & terms) / max(1, len(qset))

def search_best_chunks(session_dir: Path, query: str, k=3):
    idx = build_index_for_session(session_dir)
    if not idx:
        return []
    q_terms = tokenize(query)
    if not q_terms:
        return []
    scored = []
    for fname, ch in idx:
        s = score_chunk(q_terms, ch)
        if s > 0:
            scored.append((s, fname, ch))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:k]

# ---------- endpoints ----------
@app.get("/api/health")
def health():
    return jsonify(status="ok")

@app.post("/api/chat")
def chat():
    """
    Acepta:
    - form-data: message, session_id, files (opcional)
    - JSON: { "message": "...", "session_id": "..." }
    """
    # Soporta JSON o form
    if request.is_json:
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        session_id = data.get("session_id") or uuid.uuid4().hex
        incoming_files = []
    else:
        message = (request.form.get("message") or "").strip()
        session_id = request.form.get("session_id") or uuid.uuid4().hex
        incoming_files = request.files.getlist("files") if "files" in request.files else []

    if not message and not incoming_files:
        return jsonify(error="Envía un mensaje o adjunta un archivo."), 400

    ses_dir = ensure_session_dir(session_id)
    saved_files = []

    # Guardar adjuntos (si vienen)
    try:
        for f in incoming_files:
            if not f or not f.filename:
                continue
            dest = save_file(f, ses_dir)
            saved_files.append(dest.name)
    except ValueError as ve:
        return jsonify(error=str(ve)), 415

    # Buscar en PDFs de la sesión
    results = search_best_chunks(ses_dir, message, k=3)

    if results:
        # arma respuesta con pequeñas citas
        lines = [f"**Respuesta basada en tus documentos** (coincidencias: {len(results)})"]
        for i, (score, fname, chunk) in enumerate(results, 1):
            # recorta el fragmento
            snippet = chunk
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "…"
            lines.append(f"{i}. *{fname}* — {snippet}")
        reply = "\n\n".join(lines)
    else:
        # si no hay resultados pero hay corpus, di que no hubo coincidencias
        has_pdfs = any(p.suffix.lower() == ".pdf" for p in ses_dir.iterdir())
        if has_pdfs:
            reply = ("No encontré coincidencias claras en los PDFs para tu pregunta.\n"
                     "Prueba a usar palabras clave más concretas o sube un documento con ese contenido.")
        else:
            reply = ("No he encontrado archivos PDF en tu sesión.\n"
                     "Adjunta PDFs para que pueda responder según su contenido.")

    return jsonify(session_id=session_id,
                   reply=reply,
                   used_files=saved_files)

# Arranque para Render (usa $PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
