
import os, uuid, re, json
from pathlib import Path
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# ===== Config =====
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTS = {".pdf"}  # (si activas OCR, añade .png .jpg .jpeg)
MAX_CONTENT_MB = 25

# ===== App =====
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ===== Utilidades =====
def allowed_file(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTS

def session_dir(session_id: str) -> Path:
    d = UPLOAD_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def extract_text_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        return f"[No se pudo extraer {path.name}: {e}]"

def load_corpus(session_id: str) -> Dict[str, str]:
    """Devuelve {filename: text} de todos los PDFs extraídos en la sesión."""
    d = session_dir(session_id)
    corpus = {}
    for f in d.iterdir():
        if f.suffix.lower() == ".pdf":
            txt_path = f.with_suffix(".txt")
            if txt_path.exists():
                corpus[f.name] = txt_path.read_text(encoding="utf-8", errors="ignore")
    return corpus

def save_and_index(file_storage, ses_dir: Path) -> str:
    safe_name = secure_filename(file_storage.filename or f"file_{uuid.uuid4().hex}")
    if not allowed_file(safe_name):
        raise ValueError("Tipo de archivo no permitido")
    dest = ses_dir / safe_name
    file_storage.save(dest)

    # extrae texto y guarda .txt para búsquedas rápidas
    text = ""
    if dest.suffix.lower() == ".pdf":
        text = extract_text_pdf(dest)
    # (OCR ganchos: si activas, procesa .png/.jpg aquí)

    (ses_dir / (dest.stem + ".txt")).write_text(text, encoding="utf-8", errors="ignore")
    return safe_name

# ===== Endpoints =====
@app.get("/api/health")
def health():
    return jsonify(status="ok")

@app.post("/api/upload")
def upload():
    sid = (request.form.get("session_id") or "").strip()
    if not sid:
        return jsonify(error="Falta session_id"), 400
    if "files" not in request.files:
        return jsonify(error="Sube al menos un archivo (campo files)"), 400

    ses_dir = session_dir(sid)
    saved = []
    for f in request.files.getlist("files"):
        if not f or not f.filename:
            continue
        try:
            name = save_and_index(f, ses_dir)
            saved.append(name)
        except ValueError as ve:
            return jsonify(error=str(ve)), 415
        except Exception as e:
            return jsonify(error=f"Error guardando {f.filename}: {e}"), 500

    return jsonify(session_id=sid, saved=saved, total=len(saved))

@app.get("/api/files")
def list_files():
    sid = (request.args.get("session_id") or "").strip()
    if not sid:
        return jsonify(error="Falta session_id"), 400
    d = session_dir(sid)
    files = [p.name for p in d.iterdir() if p.suffix.lower() in {".pdf"}]
    return jsonify(session_id=sid, files=files)

# === Búsqueda básica con TF-IDF (simple, sin embeddings) ===
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def top_snippets(corpus: Dict[str, str], query: str, k: int = 3) -> List[str]:
    if not corpus:
        return []
    docs = list(corpus.values())
    keys = list(corpus.keys())
    try:
        vec = TfidfVectorizer(stop_words="spanish")
        m = vec.fit_transform(docs + [query])
        sims = cosine_similarity(m[-1], m[:-1]).flatten()
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for idx, score in ranked:
            doc = docs[idx]
            # saca un trozo “representativo”
            snippet = doc[:600].replace("\n", " ").strip()
            out.append(f"[{keys[idx]} · score {score:.2f}] {snippet}")
        return out
    except Exception:
        # fallback bruto: devuelve primeras líneas
        out = []
        for kname, doc in list(corpus.items())[:k]:
            out.append(f"[{kname}] {doc[:600].replace(chr(10),' ')}")
        return out

@app.post("/api/chat")
def chat():
    # acepta multipart/form-data o JSON; prioriza form
    message = (request.form.get("message") or request.json.get("message") if request.is_json else "").strip()
    sid = (request.form.get("session_id") or request.json.get("session_id") if request.is_json else "").strip()

    if not sid:
        sid = uuid.uuid4().hex  # crea sesión si no viene
    if not message:
        return jsonify(error="Falta message", session_id=sid), 400

    # Si adjuntaron ficheros aquí, también procesa (opcional)
    if "files" in request.files:
        ses_dir = session_dir(sid)
        for f in request.files.getlist("files"):
            if f and f.filename:
                try:
                    save_and_index(f, ses_dir)
                except Exception:
                    pass

    corpus = load_corpus(sid)
    if not corpus:
        return jsonify(
            session_id=sid,
            reply=("No tengo documentos en tu sesión. Sube PDFs con /api/upload primero "
                   "o adjúntalos en el chat. Luego pregunta y responderé con base en ellos.")
        )

    # Recuperación simple
    hits = top_snippets(corpus, message, k=3)
    if not hits:
        return jsonify(session_id=sid, reply="No encontré coincidencias en los documentos.")

    reply = "Esto es lo más relevante que encontré en tus documentos:\n\n" + "\n\n".join(hits)
    return jsonify(session_id=sid, reply=reply)

if __name__ == "__main__":
    # En producción: usa un WSGI (gunicorn/uvicorn) detrás de HTTPS
    app.run(host="0.0.0.0", port=8000, debug=True)
