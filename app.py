
"""
Caramel AI - Cybersecurity Analyst Chatbot with RAG
Built at HERE AND NOW AI – Artificial Intelligence Research Institute
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import requests
import os
import logging
import uuid
import re
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# RAG dependencies
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)
app.secret_key = '4903814210700493'  # Change in production

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================
class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    # Knowledge base path (root relative)
    KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'Data', "pdfs for knowledge base")

    # Embedding model
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')


# =========================
# Session + History
# =========================
first_interaction = True
chat_histories = {}

# =========================
# RAG: Global state
# =========================
embedding_model = None         # SentenceTransformer
faiss_index = None             # FAISS index (IndexFlatIP)
kb_chunks = []                 # List[dict]: {text, source, page}
kb_embeddings = None           # np.ndarray (N, d)
embedding_dim = None

# =========================
# Helper Functions
# =========================

CYBERSECURITY_KEYWORDS = [
    'security', 'cyber', 'hack', 'hacking', 'password', 'phishing', 'malware',
    'virus', 'ransomware', 'firewall', 'antivirus', 'encryption', 'vpn', 'ssl',
    'https', 'breach', 'attack', 'vulnerability', 'exploit', 'trojan', 'spyware',
    'adware', 'ddos', 'social engineering', 'two factor', '2fa', 'mfa', 'backup',
    'incident', 'compromise', 'threat', 'risk', 'privacy', 'authentication',
    'authorization', 'certificate', 'secure', 'protection', 'defend', 'safety',
    'safe', 'patch', 'update', 'wifi', 'network', 'router', 'endpoint',
    'identity theft', 'scam', 'fraud', 'suspicious', 'malicious', 'data breach'
]

def is_greeting(message: str) -> bool:
    greetings = [
        'hello','hi','hey','good morning','good afternoon','good evening',
        'start','begin'
    ]
    ml = message.lower().strip()
    acknowledgments = ['thank you','thanks','thx','ok','okay','tnx','ty']
    if any(ack in ml for ack in acknowledgments):
        return False
    return any(ml == greet or ml.startswith(greet) for greet in greetings)

def is_acknowledgment(message: str) -> bool:
    ml = message.lower().strip()
    return any(ack in ml for ack in ['thank you','thanks','thx','tnx','ty','ok','okay'])

def is_vague(msg: str) -> bool:
    vague_patterns = [
        'explain','tell me','what','how','about it','properly','more',
        'details','elaborate','expand','describe','yes','ok','sure','continue','why'
    ]
    ml = msg.lower().strip()
    return len(ml) < 25 and any(pat == ml or pat in ml for pat in vague_patterns)

def is_cybersecurity_related(msg: str) -> bool:
    ml = msg.lower()
    return any(kw in ml for kw in CYBERSECURITY_KEYWORDS)

def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# =========================
# RAG: Indexing and Retrieval
# =========================
def ensure_embedding_model():
    global embedding_model, embedding_dim
    if embedding_model is None:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        test_vec = embedding_model.encode(["test"], normalize_embeddings=True)
        embedding_dim = test_vec.shape[1]
        logger.info(f"Embedding model loaded. Dimension={embedding_dim}")

def pdf_to_text_chunks(pdf_path, chunk_size=700, overlap=120):
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_text(text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "source": os.path.basename(pdf_path),
                    "page": i + 1
                })
            if end == len(text):
                break
            start = max(0, end - overlap)
    return chunks

def build_kb_index():
    """Build FAISS index from PDFs in Data/pdfs for knowledge base"""
    global faiss_index, kb_chunks, kb_embeddings, embedding_dim

    ensure_embedding_model()

    kb_dir = Config.KB_DIR
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)
        logger.info(f"Knowledge base folder created at: {kb_dir} (add PDFs here)")
        faiss_index = None
        kb_chunks = []
        kb_embeddings = None
        return {"success": True, "message": f"KB folder ready. Add PDFs to: {kb_dir}", "count": 0}

    pdf_files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        faiss_index = None
        kb_chunks = []
        kb_embeddings = None
        return {"success": True, "message": f"No PDFs found in {kb_dir}", "count": 0}

    all_chunks = []
    for fpath in pdf_files:
        try:
            chunks = pdf_to_text_chunks(fpath)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")

    if not all_chunks:
        faiss_index = None
        kb_chunks = []
        kb_embeddings = None
        return {"success": False, "message": "No content extracted from PDFs.", "count": 0}

    texts = [c["text"] for c in all_chunks]
    embs = embedding_model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss_index = index
    kb_chunks = all_chunks
    kb_embeddings = embs
    return {"success": True, "message": "KB index built successfully.", "count": len(kb_chunks)}

def retrieve_context(query: str, k: int = 4, min_score: float = 0.2) -> str:
    """
    Return concatenated top-k chunks as context, or empty string if none.
    """
    if faiss_index is None or kb_embeddings is None or not kb_chunks:
        return ""

    ensure_embedding_model()
    q_emb = embedding_model.encode([query], normalize_embeddings=True)  # (1, d)
    q_emb = np.array(q_emb, dtype="float32")

    scores, idxs = faiss_index.search(q_emb, k)  # (1, k) each

    # Flatten to 1D arrays, robustly
    scores = np.asarray(scores).reshape(-1)
    idxs = np.asarray(idxs).reshape(-1)

    parts = []
    for score, idx in zip(scores, idxs):
        # Convert potential 0-d arrays to Python scalars safely
        try:
            idx_int = int(np.asarray(idx).item())
            score_f = float(np.asarray(score).item())
        except Exception:
            continue

        if idx_int == -1:
            continue
        if score_f < float(min_score):
            continue
        if 0 <= idx_int < len(kb_chunks):
            chunk = kb_chunks[idx_int]
            src = chunk.get("source", "KB")
            pg = chunk.get("page", "?")
            txt = (chunk.get("text", "") or "").strip()
            if txt:
                parts.append(f"[{src} p.{pg}] {txt}")

    return "\n\n".join(parts)

# =========================
# System Prompt (with optional RAG)
# =========================
def get_caramel_system_prompt(user_message: str, is_first: bool = False, kb_context: str = "") -> str:
    """Generate system prompt. If kb_context exists, include it under 'Company/KB Context'."""
    if is_greeting(user_message) and is_first:
        base = (
            "You are Caramel AI – a professional Cybersecurity Analyst from HERE AND NOW AI.\n"
            "For the very first greeting:\n"
            "- ALWAYS start exactly with: 'Hi, I am Caramel AI, your cybersecurity analyst from HERE AND NOW AI 🛡️.'\n"
            "- Follow with 1–2 warm sentences about how you can help with cybersecurity.\n"
            "- Encourage them to ask a cybersecurity question.\n\n"
            "Example: Hi, I am Caramel AI, your cybersecurity analyst from HERE AND NOW AI 🛡️. "
            "I help protect people against threats like phishing, ransomware, and data breaches. "
            "What cybersecurity question would you like to ask me today?"
        )

    elif is_greeting(user_message) and not is_first:
        base = (
            "You are Caramel AI – a professional Cybersecurity Analyst from HERE AND NOW AI.\n"
            "For repeat greetings:\n"
            "- Start again with: 'Hi, I am Caramel AI 🛡️.'\n"
            "- Keep it short (1–2 sentences max).\n"
            "- Stay friendly and professional.\n\n"
            "Example: Hi, I am Caramel AI 🛡️, glad to see you again! "
            "What cybersecurity area would you like to explore today?"
        )

    elif is_acknowledgment(user_message):
        base = (
            "You are Caramel AI – Cybersecurity Analyst.\n"
            "When the user thanks you or acknowledges:\n"
            "- Respond politely in 1–2 sentences\n"
            "- Do not re-introduce yourself\n"
            "- Offer further help on cybersecurity\n\n"
            "Example: \"You're most welcome! 🛡 Glad I could help. "
            "Let me know if you'd like more cybersecurity tips anytime.\""
        )
    elif is_vague(user_message):
        base = (
            "You are Caramel AI – Cybersecurity Analyst.\n"
            "For vague prompts like 'why', 'ok', 'explain more':\n"
            "- Provide a helpful follow-up in 1–2 lines without asking questions back.\n"
            "- Suggest 2–3 relevant cybersecurity topics to pick from.\n\n"
            "Example: \"Happy to help! Would you like insights on phishing risks, password safety, or malware protection? "
            "Tell me which one and I’ll explain clearly.\""
        )
    elif not is_cybersecurity_related(user_message):
        base = (
            "You are Caramel AI – AI Cybersecurity Analyst from HERE AND NOW AI.\n"
            "When user asks about a NON-CYBERSECURITY topic:\n"
            "- Give a short 2–3 sentence valid answer about their question.\n"
            "- Then politely redirect to cybersecurity expertise.\n"
            "- Friendly, professional tone.\n\n"
            "Example: The Eiffel Tower was built in 1889 as the centerpiece of the Paris World Fair. "
            "Today it is a major global landmark. "
            "By the way, my real expertise is cybersecurity 🛡. "
            "Would you like some safety tips for your digital life?"
        )
    else:
        base = (
            "You are Caramel AI – Cybersecurity Analyst built at HERE AND NOW AI.\n"
            "For cybersecurity questions:\n"
            "- Start with a 1 sentence intro\n"
            "- Provide 3–5 bullet points with clear info\n"
            "- End with 1–2 practical security tips\n"
            "- Keep response under 160 words, well-structured\n"
            "- Use emojis like 🚨 🛡 ✅ ⚠"
        )

    if kb_context:
        base += (
            "\n\nCompany/KB Context (use only if relevant; cite source filenames inline in prose):\n"
            f"{kb_context}"
        )
    return base

# =========================
# Gemini API Call (with RAG)
# =========================
def call_gemini_api(message: str) -> str:
    global first_interaction
    try:
        if not message or not isinstance(message, str):
            return "⚠ Invalid message."
        if not Config.GEMINI_API_KEY:
            logger.error("Missing GEMINI_API_KEY")
            return "🚨 Missing API key."

        # Session
        if "session_id" not in session:
            session["session_id"] = str(uuid.uuid4())
        sid = session["session_id"]
        if sid not in chat_histories:
            chat_histories[sid] = []

        # RAG
        kb_context = retrieve_context(message, k=4, min_score=0.22)
        system_prompt = get_caramel_system_prompt(message, first_interaction, kb_context)
        history = chat_histories[sid]

        # Gemini expects "contents"
        contents = [
            {"role": "user", "parts": [{"text": system_prompt}]}
        ]
        for h in history:
            contents.append({"role": h["role"], "parts": [{"text": h["content"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{Config.GEMINI_MODEL}:generateContent?key={Config.GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 4096
            }
        }

        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            logger.error(f"Gemini error {resp.status_code}: {resp.text[:300]}")
            return f"🚨 API error {resp.status_code}"

        res = resp.json()
        try:
            ai_text = res["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            logger.error(f"Gemini bad response: {res}")
            return "⚠ AI response parsing error."

        # Store history
        history.append({"role": "user", "content": message})
        history.append({"role": "model", "content": ai_text})
        if len(history) > 20:
            history = history[-20:]
        chat_histories[sid] = history

        if first_interaction:
            first_interaction = False
        return ai_text

    except Exception as e:
        logger.exception("Unexpected error in call_gemini_api")
        return "⚠ Unexpected error."


# =========================
# Flask Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data"})
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"success": False, "message": "Empty message"})
        resp = call_gemini_api(msg)   
        return jsonify({"success": True, "response": resp})
    except Exception as e:
        logger.error(str(e))
        return jsonify({"success": False, "message": "Server error"})


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    sid = session.get("session_id")
    if sid and sid in chat_histories:
        del chat_histories[sid]
    return jsonify({"success": True, "message": "Chat history cleared"})

@app.route('/api/reindex', methods=['POST'])
def api_reindex():
    """Rebuild the KB index from PDFs under Data/pdfs for knowledge base"""
    try:
        result = build_kb_index()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        return jsonify({"success": False, "message": "Failed to build index"})

@app.route('/api/debug', methods=['POST'])
def api_debug():
    try:
        data = request.get_json() or {}
        msg = data.get("message","").strip()
        kb_ready = faiss_index is not None and bool(kb_chunks)
        return jsonify({
            "has_key": bool(Config.GEMINI_API_KEY),
            "model": Config.GEMINI_MODEL,
            "session_id": session.get("session_id"),
            "history_len": len(chat_histories.get(session.get("session_id"), [])) if session.get("session_id") in chat_histories else 0,
            "is_greeting": is_greeting(msg) if msg else None,
            "is_ack": is_acknowledgment(msg) if msg else None,
            "is_vague": is_vague(msg) if msg else None,
            "is_cyber": is_cybersecurity_related(msg) if msg else None,
            "kb_ready": kb_ready,
            "kb_dir": Config.KB_DIR,
            "kb_chunks": len(kb_chunks) if kb_chunks else 0
        })
    except Exception as e:
        logger.error(f"Debug error: {e}")
        return jsonify({"error":"debug failed"}), 500

@app.route('/health')
def health():
    kb_ready = faiss_index is not None and kb_chunks
    return jsonify({
        "status": "healthy",
        "service": "Caramel AI Cybersecurity Analyst",
        "kb_indexed": bool(kb_ready),
        "kb_chunks": len(kb_chunks) if kb_chunks else 0,
        "kb_path": Config.KB_DIR
    })

if __name__ == "__main__":
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    # Ensure KB directory exists
    os.makedirs(Config.KB_DIR, exist_ok=True)

    # Optional: build index on startup
    try:
        build_result = build_kb_index()
        logger.info(f"KB build: {build_result}")
    except Exception as e:
        logger.error(f"KB build on startup failed: {e}")

    print("🛡 Caramel AI with RAG started at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)