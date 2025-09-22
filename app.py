
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# --- Config ---
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


# ---------- Utilidades de ficheros / sesiones ----------
def allowed_file(name_or_path) -> bool:
    return Path(name_or_path).suffix.lower() in ALLOWED_EXTS

def ensure_session_dir(session_id: str) -> Path:
    ses = UPLOAD_ROOT / session_id
    ses.mkdir(parents=True, exist_ok=True)
    return ses

def save_file(file_storage, dest_dir: Path) -> Path:
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}")
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido (usa .pdf/.png/.jpg/.jpeg).")
    dest = dest_dir / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_storage.save(dest)
    return dest

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        buff = []
        for page in reader.pages:
            try:
                buff.append(page.extract_text() or "")
            except Exception:
                buff.append("")
        return "\n".join(buff)
    except Exception as e:
        return f"[No se pudo extraer texto de {pdf_path.name}: {e}]"

def collect_session_texts(ses_dir: Path) -> List[Tuple[str, str]]:
    """
    Devuelve [(nombre_fichero, texto)] solo de PDFs.
    (Si quisieras OCR para imágenes, aquí sería el sitio.)
    """
    out = []
    for p in ses_dir.iterdir():
        if p.suffix.lower() == ".pdf":
            out.append((p.name, extract_text_from_pdf(p)))
    return out


# ---------- “NLP” ligero ----------
STOPWORDS = set("""
a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el
ella ellas ellos en entre era eran es esa esas ese eso esos esta estaba estaban estamos estar este
estos fue fueron ha hasta hay la las le les lo los mas me mi mis mucho muy nada ni no nos o os
otra otros para pero poco por porque que quien quienes se ser si sobre sin su sus te tiene tuvo un
una uno y ya yo tu usted ustedes él ella ello nosotros vosotros
kg ml mg mcg min hora horas h l/min
""".split())

def tokenize(text: str) -> List[str]:
    # minúsculas + palabras
    return re.findall(r"[a-záéíóúüñ0-9]+", text.lower())

def key_terms(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]

def split_sentences(text: str) -> List[str]:
    # Segmentación simple por puntos/interrogaciones/exclamaciones y saltos de línea
    parts = re.split(r"[\.!\?\n]+", text)
    return [p.strip() for p in parts if p.strip()]

def score_sentence(sentence: str, q_terms: List[str]) -> int:
    s_terms = set(key_terms(sentence))
    return sum(1 for t in q_terms if t in s_terms)

def extractive_answer(docs: List[Tuple[str, str]], question: str, top_k: int = 3):
    q_terms = key_terms(question)
    if not q_terms:
        return None

    candidates = []  # (score, sentence, filename)
    for fname, fulltext in docs:
        for sent in split_sentences(fulltext):
            sc = score_sentence(sent, q_terms)
            if sc > 0:
                candidates.append((sc, sent, fname))

    if not candidates:
        return None

    # ordenar por score (desc) y longitud (desc ligera para evitar frases muy cortas)
    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    top = candidates[:top_k]

    hits = "\n".join([f"• {s}  ({fn})" for _, s, fn in top])
    return hits


# ---------- Respuestas de “chat ligero” ----------
def smalltalk_reply(user_text: str) -> str:
    t = user_text.strip().lower()

    # saludos
    if re.search(r"\b(hola|buenas|buenos días|buenas tardes|hey|qué tal)\b", t):
        return "¡Hola! Soy tu asistente UMAER 😊. Puedo charlar y también buscar en los PDF que hayas cargado. ¿En qué te ayudo?"

    # despedidas
    if re.search(r"\b(ad[ií]os|hasta luego|nos vemos|gracias,? ad[ií]os)\b", t):
        return "¡Hasta luego! Si necesitas algo, aquí estaré. 👋"

    # quién eres
    if re.search(r"qu[ié]n eres|qu[ée] puedes hacer|ayuda|help", t):
        return (
            "Soy un asistente ‘ligero’: puedo conversar de forma natural y, si adjuntas o ya tienes PDFs en el gestor, "
            "puedo buscar información relevante en ellos para responder mejor. "
            "Prueba con: “resume el protocolo X del PDF”, “¿qué dosis sugiere el documento Y?”, etc."
        )

    # gracias
    if re.search(r"\b(gracias|muchas gracias|mil gracias)\b", t):
        return "¡A ti! ¿Seguimos con algo más? 🙌"

    # fallback general
    return (
        "Te escucho. Si tu pregunta depende de documentos, adjunta un PDF o usa el gestor de archivos, "
        "y lo tendré en cuenta. Si es charla general, también puedo responder de forma natural 🙂."
    )


# ---------- Rutas ----------
@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.post("/api/chat")
def chat():
    """
    Acepta:
    - multipart/form-data: fields 'message', 'session_id', y opcional 'files'
    - application/json:    {"message": "...", "session_id": "..."}
    """
    # 1) Leer message + session_id desde multipart o JSON
    message = ""
    session_id = request.form.get("session_id") or ""
    if request.content_type and "multipart/form-data" in request.content_type.lower():
        message = (request.form.get("message") or "").strip()
    else:
        try:
            js = request.get_json(silent=True) or {}
        except Exception:
            js = {}
        message = (js.get("message") or "").strip()
        session_id = js.get("session_id") or session_id

    if not session_id:
        session_id = uuid.uuid4().hex

    ses_dir = ensure_session_dir(session_id)

    # 2) Guardar archivos si vienen en multipart
    saved_files = []
    if request.content_type and "multipart/form-data" in request.content_type.lower():
        if "files" in request.files:
            for f in request.files.getlist("files"):
                if f and f.filename:
                    try:
                        dest = save_file(f, ses_dir)
                        saved_files.append(dest.name)
                    except ValueError as ve:
                        return jsonify(error=str(ve)), 415

    # 3) Cargar corpus de la sesión
    docs = collect_session_texts(ses_dir)

    # 4) Lógica de respuesta
    if not message:
        # Sin mensaje: devolver estado de sesión y ficheros
        return jsonify(
            session_id=session_id,
            reply=(
                "Sesión iniciada. Puedes escribir un mensaje o adjuntar un PDF.\n"
                f"Archivos actuales: {', '.join([n for n, _ in docs]) if docs else '(ninguno)'}"
            ),
            used_files=saved_files,
            mode="status"
        )

    # Si hay documentos, intentar respuesta extractiva
    if docs:
        hits = extractive_answer(docs, message, top_k=3)
        if hits:
            reply = (
                "Esto es lo más relevante que he encontrado en tus documentos:\n\n"
                f"{hits}\n\n"
                "¿Quieres que lo resuma o que busque algo más específico?"
            )
            return jsonify(session_id=session_id, reply=reply, used_files=saved_files, mode="doc_search")

        # No hubo coincidencias claras → responder con chat ligero + sugerencias
        base = smalltalk_reply(message)
        reply = (
            f"{base}\n\n"
            "No encontré coincidencias claras en los PDFs actuales. Si quieres, dime palabras clave más concretas "
            "o adjunta el documento donde conste."
        )
        return jsonify(session_id=session_id, reply=reply, used_files=saved_files, mode="chat_fallback")

    # No hay documentos → responder en modo “chat”
    reply = smalltalk_reply(message)
    return jsonify(session_id=session_id, reply=reply, used_files=saved_files, mode="chat")


if __name__ == "__main__":
    # En Render usan gunicorn normalmente; esto es útil en local.
    app.run(host="0.0.0.0", port=8000, debug=True)
