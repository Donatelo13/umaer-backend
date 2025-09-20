
import os
import uuid
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# ====== Config ======
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf"}   # seguimos aceptando sólo PDF para búsqueda (imágenes requerirían OCR)
MAX_CONTENT_MB = 25

# ====== App ======
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
CORS(app)


# ====== Utilidades ======
def allowed_file(name_or_path) -> bool:
    return Path(name_or_path).suffix.lower() in ALLOWED_EXTS


def ensure_session_dir(session_id: str) -> Path:
    ses_dir = UPLOAD_ROOT / session_id
    ses_dir.mkdir(parents=True, exist_ok=True)
    return ses_dir


def save_file(file_storage, dest_dir: Path) -> Path:
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}.pdf")
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido (solo .pdf por ahora)")
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    """
    Devuelve una lista con el texto de cada página del PDF (índice 0 = página 1).
    Si no se puede extraer, devuelve lista vacía.
    """
    try:
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        return pages_text
    except Exception:
        return []


def normalize_text(s: str) -> str:
    # minúsculas + quitar acentos para comparación robusta
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def tokenize_query(q: str) -> List[str]:
    qn = normalize_text(q)
    terms = re.findall(r"[a-z0-9áéíóúüñ]+", qn)  # ya normalizado, pero admitimos caracteres alfanum
    # Filtra términos muy cortos que suelen meter ruido
    return [t for t in terms if len(t) >= 2]


def score_and_snippets(page_text: str, terms: List[str], snippet_len: int = 220) -> Dict[str, Any]:
    """
    Calcula un score sencillo (frecuencia total de términos) y crea un snippet
    alrededor de la primera coincidencia más significativa.
    """
    if not page_text.strip():
        return {"score": 0, "snippet": ""}

    norm_page = normalize_text(page_text)

    # conteo simple de apariciones
    score = 0
    first_hit_index = None
    for t in terms:
        for m in re.finditer(re.escape(t), norm_page):
            score += 1
            if first_hit_index is None:
                first_hit_index = m.start()

    if score == 0:
        return {"score": 0, "snippet": ""}

    # construir snippet alrededor de la primera coincidencia
    i = max(0, first_hit_index - snippet_len // 2)
    j = min(len(page_text), i + snippet_len)
    snippet = page_text[i:j].strip()

    # embellecer: cortar en límites de palabra y añadir “…”
    if i > 0:
        snippet = "…" + snippet
    if j < len(page_text):
        snippet = snippet + "…"

    return {"score": score, "snippet": snippet}


def search_in_session(session_dir: Path, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Busca los términos de 'query' en todos los PDF de la sesión.
    Devuelve una lista con los mejores resultados: file, page, score, snippet.
    """
    terms = tokenize_query(query)
    if not terms:
        return []

    results = []
    for pdf in sorted(session_dir.glob("*.pdf")):
        pages = extract_pdf_pages(pdf)
        for idx, page_text in enumerate(pages):
            res = score_and_snippets(page_text, terms)
            if res["score"] > 0:
                results.append({
                    "file": pdf.name,
                    "page": idx + 1,  # humano: empieza en 1
                    "score": res["score"],
                    "snippet": res["snippet"]
                })

    # ordena por score descendente y recorta
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def collect_first_chars(session_dir: Path, limit: int = 500) -> str:
    """
    Como fallback, toma los primeros 'limit' caracteres del primer PDF con texto.
    """
    for pdf in sorted(session_dir.glob("*.pdf")):
        pages = extract_pdf_pages(pdf)
        if pages:
            text = "\n".join(pages).strip()
            if text:
                return text[:limit]
    return ""


# ====== Endpoints ======
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
    # Guardar archivos (si llegan)
    if "files" in request.files:
        for f in request.files.getlist("files"):
            if not f or not f.filename:
                continue
            try:
                dest = save_file(f, ses_dir)
                saved_files.append(dest.name)
            except ValueError as ve:
                return jsonify(error=str(ve)), 415

    # Si hay pregunta, buscar en PDFs
    reply_text: str
    matches: List[Dict[str, Any]] = []
    if message:
        matches = search_in_session(ses_dir, message, top_k=5)

    if matches:
        # Construimos una respuesta legible con top resultados
        lines = ["He buscado en tus documentos y esto es lo más relevante:\n"]
        for m in matches:
            lines.append(f"• {m['file']} — pág. {m['page']} (relevancia {m['score']}):\n  {m['snippet']}\n")
        # Un consejo para afinar
        lines.append("Consejo: puedes refinar la pregunta con más detalle para obtener fragmentos más específicos.")
        reply_text = "\n".join(lines)
    else:
        # Sin matches: fallback informativo
        if any(ses_dir.glob("*.pdf")):
            preview = collect_first_chars(ses_dir, limit=500)
            reply_text = (
                "No encontré coincidencias directas para tu pregunta en los PDFs. "
                "Prueba con palabras clave distintas (p. ej., términos exactos que aparezcan en el documento).\n\n"
                "Vista previa del contenido (primeros 500 caracteres del primer PDF con texto):\n"
                f"{preview or '[No se pudo extraer texto]'}"
            )
        else:
            reply_text = (
                "No hay PDFs en tu sesión todavía. Adjunta uno (PDF) y vuelve a intentar tu pregunta."
            )

    return jsonify(
        session_id=session_id,
        used_files=saved_files,
        matches=matches,     # opcional, por si luego quieres mostrar lista clicable
        reply=reply_text
    )


if __name__ == "__main__":
    # Si estás en Render u otro PaaS, ignora este run local.
    app.run(host="0.0.0.0", port=8000, debug=True)
