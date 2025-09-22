
import os
import uuid
import re
import json
from pathlib import Path
from typing import List, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# =============== Config básica ===============
# En Render, usa /tmp (persistencia efímera, válida para la sesión del proceso)
BASE_DIR = Path("/tmp")
UPLOAD_ROOT = BASE_DIR / "uploads"
SESSIONS_ROOT = BASE_DIR / "sessions"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".pdf"}  # si quieres permitir imágenes, añádelas, pero aquí solo PDF
MAX_UPLOAD_MB = 25

# OpenAI opcional
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # cambia si prefieres otro
USE_OPENAI = bool(OPENAI_API_KEY)

# Flask
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024  # 25 MB

# =============== Utilidades de ficheros ===============
def allowed_file(name_or_path) -> bool:
    return Path(name_or_path).suffix.lower() in ALLOWED_EXTS

def ensure_session_dir(session_id: str) -> Path:
    d = UPLOAD_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def session_state_path(session_id: str) -> Path:
    return SESSIONS_ROOT / f"{session_id}.json"

def load_session_state(session_id: str) -> dict:
    p = session_state_path(session_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_session_state(session_id: str, data: dict):
    p = session_state_path(session_id)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def save_file(file_storage, dest_dir: Path) -> Path:
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}")
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido (solo PDF).")
    dest = dest_dir / safe_name
    file_storage.save(dest)
    return dest

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        return f"[Error leyendo {pdf_path.name}: {e}]"

# =============== Indexado y búsqueda (RAG-lite) ===============
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+")

def tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s or "")]

def chunk_text(text: str, size=900, overlap=150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        # intenta cortar en fin de frase si queda cerca
        if end < n:
            p = text.find(".", end, min(n, end + 200))
            if p != -1:
                chunk = text[start:p + 1]
                end = p + 1
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks

def build_index_for_session(session_dir: Path) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (filename, chunk_text) para todos los PDFs en la sesión.
    """
    index = []
    for p in sorted(session_dir.iterdir()):
        if p.suffix.lower() == ".pdf":
            full = extract_text_from_pdf(p)
            for ch in chunk_text(full):
                if ch:
                    index.append((p.name, ch))
    return index

def score_chunk(query_terms: set, chunk: str) -> float:
    terms = set(tokenize(chunk))
    if not terms:
        return 0.0
    # métrica simple: intersección normalizada
    return len(query_terms & terms) / max(1, len(query_terms))

def search_best_chunks(session_dir: Path, query: str, k=4) -> List[Tuple[float, str, str]]:
    idx = build_index_for_session(session_dir)
    if not idx:
        return []
    q_terms = set(tokenize(query))
    if not q_terms:
        return []
    scored = []
    for fname, ch in idx:
        s = score_chunk(q_terms, ch)
        if s > 0:
            scored.append((s, fname, ch))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:k]

# =============== OpenAI (opcional) ===============
def call_openai(messages: List[dict], max_tokens=600, temperature=0.3) -> str:
    """
    Llama al modelo de OpenAI si hay API key. Si no, devuelve cadena vacía.
    """
    if not USE_OPENAI:
        return ""
    try:
        # SDK ligero vía HTTP para evitar versiones
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error del modelo: {e}]"

# =============== Helpers conversación ===============
SYSTEM_PROMPT = (
    "Eres el asistente médico de UMAER. Responde en español, claro y conciso. "
    "Si proporciono fragmentos de documentos, ÚSALOS como base principal. "
    "Cita el nombre del archivo entre paréntesis cuando corresponda. "
    "Si no hay información relevante en los documentos, responde de forma general, "
    "pero indica que no se halló información específica en los PDFs."
)

def build_messages(user_message: str, rag_snippets: List[Tuple[float, str, str]], history: List[dict]) -> List[dict]:
    """
    Construye el prompt para OpenAI con: system + contexto (si lo hay) + historial + mensaje del usuario.
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Inyecta contexto si hay
    if rag_snippets:
        ctx_lines = ["Contexto de documentos (extractos relevantes):"]
        for score, fname, ch in rag_snippets:
            sn = ch if len(ch) <= 700 else ch[:700].rsplit(" ", 1)[0] + "…"
            ctx_lines.append(f"- [{fname}] {sn}")
        msgs.append({"role": "system", "content": "\n".join(ctx_lines)})

    # Historial breve (últimas 6 intervenciones)
    for m in history[-6:]:
        msgs.append(m)

    # Mensaje del usuario
    msgs.append({"role": "user", "content": user_message})
    return msgs

# =============== Endpoints ===============
@app.get("/api/health")
def health():
    return jsonify(status="ok", use_openai=USE_OPENAI, model=OPENAI_MODEL if USE_OPENAI else None)

@app.post("/api/chat")
def chat():
    """
    Acepta:
    - form-data: message, session_id, files (opcional)
    - JSON: { "message": "...", "session_id": "..." }
    """
    # Entrada: JSON o form-data
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
        return jsonify(error="Envía un mensaje o adjunta un PDF."), 400

    # Directorio de sesión y guardado de adjuntos
    ses_dir = ensure_session_dir(session_id)
    saved_files = []
    try:
        for f in incoming_files:
            if f and f.filename:
                dest = save_file(f, ses_dir)
                saved_files.append(dest.name)
    except ValueError as ve:
        return jsonify(error=str(ve)), 415

    # Estado de sesión (historial de chat)
    state = load_session_state(session_id)
    history = state.get("history", [])  # lista de dicts: {"role": "user"/"assistant", "content": "..."}
    # Registramos el mensaje del usuario en historial
    if message:
        history.append({"role": "user", "content": message})

    # Búsqueda en PDFs
    rag = search_best_chunks(ses_dir, message, k=4)

    # Si hay OpenAI: conversa + usa contexto. Si no, responde extractivo.
    if USE_OPENAI:
        msgs = build_messages(message, rag, history)
        answer = call_openai(msgs)
        reply = answer.strip() if answer else "No pude generar respuesta."

        # Guarda respuesta en historial
        history.append({"role": "assistant", "content": reply})
        state["history"] = history
        save_session_state(session_id, state)
        return jsonify(session_id=session_id, reply=reply, used_files=saved_files)
    else:
        # Modo fallback (sin OpenAI): responde con citas de los documentos
        if rag:
            lines = [f"**Respuesta basada en tus documentos** (coincidencias: {len(rag)})"]
            for i, (score, fname, chunk) in enumerate(rag, 1):
                snippet = chunk if len(chunk) <= 500 else chunk[:500].rsplit(" ", 1)[0] + "…"
                lines.append(f"{i}. ({fname}) {snippet}")
            reply = "\n\n".join(lines)
        else:
            has_pdfs = any(p.suffix.lower() == ".pdf" for p in ses_dir.iterdir())
            reply = ("No encontré coincidencias en tus PDFs." if has_pdfs
                     else "No hay PDFs en tu sesión. Adjunta alguno para buscar contenido.")
        history.append({"role": "assistant", "content": reply})
        state["history"] = history
        save_session_state(session_id, state)
        return jsonify(session_id=session_id, reply=reply, used_files=saved_files)

# Gunicorn/Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
