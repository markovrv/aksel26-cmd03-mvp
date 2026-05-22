"""
Voice Streaming Server (HTTPS/WSS) — распознавание речи через Whisper
===================================
Запустить: python fserver.py
Зависимости: pip install websockets faster-whisper numpy httpx
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import ssl
import uuid
from datetime import datetime

# Windows: добавляем пути к DLL библиотекам nvidia/cuda (cublas64_12.dll и др.)
if os.name == "nt":
    _venv_base = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
    # Пути из run_server.bat
    _nv_paths = [
        os.path.join(_venv_base, "nvidia", "cublas", "bin"),
        os.path.join(_venv_base, "nvidia", "cudnn", "bin"),
        os.path.join(_venv_base, "nvidia", "cuda_nvrtc", "bin"),
    ]
    for _p in _nv_paths:
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
    # Путь к OpenSSL из run_server.bat
    _openssl_path = r"C:\Program Files\OpenSSL-Win64\bin"
    if os.path.isdir(_openssl_path):
        os.add_dll_directory(_openssl_path)

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("voice-server")

# ── Конфигурация ──────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8765
MODEL_SIZE = "medium"        # tiny | base | small | medium | large-v3
DEVICE = "cuda"               # cpu | cuda
COMPUTE_TYPE = "float16"      # int8 (CPU), float16 (GPU)
LANGUAGE = None               # язык распознавания (None — автоопределение)
SAMPLE_RATE = 16000
CHUNK_BYTES = SAMPLE_RATE * 2 * 1          # 1 секунда, 16-bit PCM, моно
SILENCE_THRESHOLD = 2000
SILENCE_CHUNKS = 2
TRANSCRIPTS_DIR = "transcripts"            # папка для сохранения стенограмм
# ──────────────────────────────────────────────────────────────

# ── Загрузка конфигурации из .env ────────────────────────────
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val

load_env()
# ──────────────────────────────────────────────────────────────

# ── Подключение к внешнему LLM (OpenAI-совместимый) ──────────
LLM_API_URL = os.environ.get("LLM_API_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
# ─────────────────────────────────────────────────────────────

# ── Настройки анти-галлюцинаций ─────────────────────────────
NO_SPEECH_THRESHOLD = 0.85
WHISPER_HALLUCINATION_BLACKLIST = [
    "thank you.", "thank you. thank you.", "thank you very much.",
    "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "you", "thanks for watching please subscribe and hit that like button...."
]
# ─────────────────────────────────────────────────────────────

# ---------- Создание папки для стенограмм ----------
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ---------- Генерация самоподписанного сертификата ----------
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

def generate_self_signed_cert():
    """Генерирует самоподписанный сертификат через openssl (если его нет)."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        log.info("Сертификат уже существует.")
        return True
    log.info("Генерация самоподписанного сертификата...")
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        log.error("OpenSSL не найден в системе. Установите openssl или создайте сертификаты вручную.")
        return False
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", KEY_FILE, "-out", CERT_FILE,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        log.info(f"Сертификат создан: {CERT_FILE}, {KEY_FILE}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Ошибка генерации сертификата: {e.stderr.decode()}")
        return False

# ====================================================================
#  Whisper-распознаватель
# ====================================================================
class WhisperRecognizer:
    def __init__(self):
        from faster_whisper import WhisperModel
        log.info(f"Загружаю модель Whisper '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info("Whisper-модель загружена.")

    def transcribe(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio,
            language=LANGUAGE,
            beam_size=5,
            no_speech_threshold=NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,
        )
        segment_texts = [seg.text.strip() for seg in segments if seg.text.strip()]
        raw_text = " ".join(segment_texts)
        filtered_text = self.filter_hallucinations(raw_text)
        log.info(f"Язык: {info.language} ({info.language_probability:.0%})  |  '{filtered_text}'")
        return filtered_text

    @staticmethod
    def filter_hallucinations(text: str) -> str:
        if not text:
            return ""
        text_lower = text.strip().lower()
        words = text_lower.split()
        if text_lower in WHISPER_HALLUCINATION_BLACKLIST:
            log.info(f"Галлюцинация удалена (полное совпадение): '{text}'")
            return ""
        if len(words) <= 3 and all(word in WHISPER_HALLUCINATION_BLACKLIST for word in words):
            log.info(f"Галлюцинация удалена (слова-паразиты): '{text}'")
            return ""
        if WhisperRecognizer._is_hallucination_by_context(text_lower):
            log.info(f"Галлюцинация удалена (контекст): '{text}'")
            return ""
        return text

    @staticmethod
    def _is_hallucination_by_context(text_lower: str) -> bool:
        hallucination_phrases = [
            "спасибо за внимание", "до свидания", "на этом всё", "конец записи"
        ]
        if any(phrase in text_lower for phrase in hallucination_phrases):
            words = text_lower.split()
            if len(words) <= 5:
                return True
        return False


# ====================================================================
#  Инициализация распознавателя
# ====================================================================
recognizer = WhisperRecognizer()

def transcribe_audio(pcm_bytes: bytes) -> str:
    return recognizer.transcribe(pcm_bytes)

def rms(pcm_bytes: bytes) -> float:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

# ---------- Работа с файлами стенограмм ----------
def list_transcript_files() -> list:
    files = []
    if not os.path.isdir(TRANSCRIPTS_DIR):
        return files
    for fname in sorted(os.listdir(TRANSCRIPTS_DIR), reverse=True):
        fpath = os.path.join(TRANSCRIPTS_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith(".txt"):
            stat = os.stat(fpath)
            files.append({
                "name": fname,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return files

def read_transcript_file(filename: str) -> str | None:
    fpath = os.path.join(TRANSCRIPTS_DIR, filename)
    if not os.path.realpath(fpath).startswith(os.path.realpath(TRANSCRIPTS_DIR)):
        return None
    if not os.path.isfile(fpath):
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

def delete_transcript_file(filename: str) -> bool:
    fpath = os.path.join(TRANSCRIPTS_DIR, filename)
    if not os.path.realpath(fpath).startswith(os.path.realpath(TRANSCRIPTS_DIR)):
        return False
    if not os.path.isfile(fpath):
        return False
    try:
        os.remove(fpath)
        return True
    except Exception:
        return False

def save_transcript_to_file(text: str, prefix: str = "transcript") -> str | None:
    if not text.strip():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    fname = f"{prefix}_{timestamp}_{unique_id}.txt"
    fpath = os.path.join(TRANSCRIPTS_DIR, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        log.info(f"Стенограмма сохранена: {fpath}")
        return fname
    except Exception as e:
        log.error(f"Ошибка сохранения стенограммы: {e}")
        return None

# ---------- LLM API (OpenAI-совместимый) ──────────────────────
def call_llm_sync(messages: list) -> dict:
    if not HAS_HTTPX:
        return {"error": "Библиотека httpx не установлена. Выполните: pip install httpx"}
    if not LLM_API_KEY:
        return {"error": "LLM_API_KEY не настроен в конфигурации сервера"}
    url = f"{LLM_API_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            return {"response": content}
    except httpx.HTTPStatusError as e:
        log.error(f"LLM HTTP ошибка: {e.response.status_code} {e.response.text[:200]}")
        return {"error": f"LLM вернул ошибку {e.response.status_code}"}
    except httpx.RequestError as e:
        log.error(f"LLM ошибка соединения: {e}")
        return {"error": f"Не удалось подключиться к LLM: {e}"}
    except Exception as e:
        log.error(f"LLM неизвестная ошибка: {e}")
        return {"error": str(e)}

# ---------- HTTP/HTTPS обработчик с API-маршрутами ----------
HTML_PATH = os.path.join(os.path.dirname(__file__), "voice-recorder.html")

def make_json_response(data: dict, status: int = 200) -> Response:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = Headers([
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-cache"),
        ("Access-Control-Allow-Origin", "*"),
    ])
    return Response(status, "OK" if status == 200 else "Error", headers, body)

def make_html_response(body: bytes, status: int = 200) -> Response:
    ct = "text/html; charset=utf-8"
    headers = Headers([
        ("Content-Type", ct),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-cache"),
    ])
    return Response(status, "OK" if status == 200 else "Error", headers, body)

def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    path = request.path.rstrip("/")
    log.info(f"HTTP {path} от {request.headers.get('Host', '?')}")

    if path == "/api/chat":
        try:
            body = request.body or b""
            data = json.loads(body)
            messages = data.get("messages", [])
            if not messages:
                return make_json_response({"error": "Поле 'messages' обязательно"}, 400)
            log.info(f"LLM запрос: {len(messages)} сообщений, модель={LLM_MODEL}")
            result = call_llm_sync(messages)
            if "error" in result:
                return make_json_response(result, 500)
            return make_json_response(result)
        except json.JSONDecodeError:
            return make_json_response({"error": "Некорректный JSON"}, 400)
        except Exception as e:
            log.error(f"Ошибка обработки /api/chat: {e}")
            return make_json_response({"error": str(e)}, 500)

    if path == "/api/health":
        return make_json_response({"status": "ok", "llm_configured": bool(LLM_API_KEY)})

    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace("ws://", "wss://")
        body = html_content.encode("utf-8")
    except FileNotFoundError:
        body = "<h1>Voice Stream</h1><p>404 HTML not found</p>".encode("utf-8")
    except Exception as e:
        log.error(f"Ошибка чтения HTML: {e}")
        body = "<h1>Voice Stream</h1><p>500 HTML reading error</p>".encode("utf-8")
    return make_html_response(body)

# ---------- Обработчик WebSocket ----------
class ClientSession:
    """Хранит состояние сессии клиента."""
    def __init__(self):
        self.buffer = bytearray()
        self.silence_count = 0
        self.current_text = ""
        self.session_id = uuid.uuid4().hex[:12]
        self.reconnected = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_name = f"live_{timestamp}_{self.session_id}.txt"
        self.file_path = os.path.join(TRANSCRIPTS_DIR, self.file_name)
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write("")
            log.info(f"Начата запись стенограммы: {self.file_path}")
        except Exception as e:
            log.error(f"Не удалось создать файл стенограммы: {e}")
            self.file_path = None

    def append_to_file(self, text: str):
        if not text.strip() or not self.file_path:
            return
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {text}\n")
            log.info(f"Стенограмма дописана: {self.file_name}")
        except Exception as e:
            log.error(f"Ошибка записи стенограммы: {e}")

async def handle_client(ws):
    """Обработка WebSocket-клиента с Whisper."""
    addr = ws.remote_address
    session = ClientSession()

    await ws.send(json.dumps({"type": "session", "session_id": session.session_id}))
    await ws.send(json.dumps({"type": "engine", "engine": "whisper"}))

    try:
        async for message in ws:
            if isinstance(message, bytes):
                session.buffer.extend(message)
                level = rms(message)
                is_silent = level < SILENCE_THRESHOLD
                if is_silent:
                    session.silence_count += 1
                else:
                    session.silence_count = 0
                await ws.send(json.dumps({"type": "level", "rms": round(level)}))

                if len(session.buffer) >= CHUNK_BYTES * 3 and session.silence_count >= SILENCE_CHUNKS:
                    audio_snapshot = bytes(session.buffer)
                    session.buffer.clear()
                    session.silence_count = 0
                    await ws.send(json.dumps({"type": "status", "text": "Распознаю..."}))
                    try:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, transcribe_audio, audio_snapshot
                        )
                        if text:
                            session.current_text += text + "\n"
                            session.append_to_file(text)
                            await ws.send(json.dumps({"type": "transcript", "text": text}))
                        else:
                            await ws.send(json.dumps({"type": "status", "text": "Готов (тишина)"}))
                    except Exception as e:
                        log.error(f"Ошибка распознавания: {e}", exc_info=True)
                        session.buffer.clear()
                        session.silence_count = 0
                        await ws.send(json.dumps({"type": "error", "text": f"Ошибка распознавания: {e}"}))

            elif isinstance(message, str):
                try:
                    cmd = json.loads(message)
                    action = cmd.get("action")

                    if action == "flush":
                        if len(session.buffer) >= SAMPLE_RATE:
                            audio_snapshot = bytes(session.buffer)
                            session.buffer.clear()
                            session.silence_count = 0
                            await ws.send(json.dumps({"type": "status", "text": "Распознаю..."}))
                            try:
                                text = await asyncio.get_event_loop().run_in_executor(
                                    None, transcribe_audio, audio_snapshot
                                )
                                if text:
                                    session.current_text += text + "\n"
                                    session.append_to_file(text)
                                    await ws.send(json.dumps({"type": "transcript", "text": text}))
                                else:
                                    await ws.send(json.dumps({"type": "status", "text": "Готов"}))
                            except Exception as e:
                                log.error(f"Ошибка распознавания (flush): {e}", exc_info=True)
                                session.buffer.clear()
                                session.silence_count = 0
                                await ws.send(json.dumps({"type": "error", "text": f"Ошибка распознавания: {e}"}))
                        else:
                            session.buffer.clear()
                        await ws.send(json.dumps({"type": "status", "text": "Готов"}))

                    elif action == "save_transcript":
                        fname = save_transcript_to_file(session.current_text)
                        if fname:
                            await ws.send(json.dumps({
                                "type": "file_saved", "filename": fname, "text": "Стенограмма сохранена"
                            }))
                        else:
                            await ws.send(json.dumps({"type": "error", "text": "Нет текста для сохранения"}))

                    elif action == "list_transcripts":
                        files = list_transcript_files()
                        await ws.send(json.dumps({"type": "transcript_list", "files": files}))

                    elif action == "get_transcript":
                        filename = cmd.get("filename", "")
                        content = read_transcript_file(filename)
                        if content is not None:
                            await ws.send(json.dumps({
                                "type": "transcript_content", "filename": filename, "content": content
                            }))
                        else:
                            await ws.send(json.dumps({"type": "error", "text": f"Файл '{filename}' не найден"}))

                    elif action == "delete_transcript":
                        filename = cmd.get("filename", "")
                        if delete_transcript_file(filename):
                            await ws.send(json.dumps({"type": "file_deleted", "filename": filename}))
                        else:
                            await ws.send(json.dumps({"type": "error", "text": f"Не удалось удалить '{filename}'"}))

                    elif action == "reconnect":
                        session.reconnected = True
                        log.info(f"Клиент {addr} переподключился (сессия {session.session_id})")
                        await ws.send(json.dumps({"type": "reconnect_ack", "session_id": session.session_id}))

                    elif action == "chat":
                        messages = cmd.get("messages", [])
                        if not messages:
                            await ws.send(json.dumps({"type": "error", "text": "Поле 'messages' обязательно"}))
                        else:
                            log.info(f"LLM запрос: {len(messages)} сообщений, модель={LLM_MODEL}")
                            try:
                                result = await asyncio.get_event_loop().run_in_executor(
                                    None, call_llm_sync, messages
                                )
                                if "error" in result:
                                    await ws.send(json.dumps({"type": "chat_response", "error": result["error"]}))
                                else:
                                    await ws.send(json.dumps({
                                        "type": "chat_response", "response": result["response"]
                                    }))
                            except Exception as e:
                                log.error(f"LLM ошибка: {e}")
                                await ws.send(json.dumps({"type": "chat_response", "error": str(e)}))

                except json.JSONDecodeError:
                    pass

    except websockets.exceptions.ConnectionClosed as e:
        log.info(f"[Whisper] Клиент отключился: {addr} (код {e.code}) (сессия {session.session_id})")
    except Exception as e:
        log.error(f"[Whisper] Ошибка соединения {addr}: {e}", exc_info=True)


# ---------- Запуск сервера с SSL ----------
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

async def main():
    if not generate_self_signed_cert():
        log.error("Не удалось получить сертификат. Завершение работы.")
        return
    local_ip = get_local_ip()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERT_FILE, KEY_FILE)
    async with websockets.serve(
        handle_client, HOST, PORT,
        process_request=process_request, ssl=ssl_context
    ):
        log.info(f"WebSocket-сервер (WSS) запущен на wss://{local_ip}:{PORT}")
        log.info(f"HTML-интерфейс доступен по адресу https://{local_ip}:{PORT}")
        log.info(f"Стенограммы сохраняются в папку '{TRANSCRIPTS_DIR}/'")
        log.info(f"Движок распознавания: Whisper")
        if LLM_API_KEY:
            log.info(f"LLM API настроен: {LLM_API_URL}, модель={LLM_MODEL}")
        else:
            log.info("LLM API не настроен. Укажите LLM_API_KEY в конфигурации.")
        log.info("Ожидаю подключений со смартфона...")
        log.warning("При использовании самоподписанного сертификата браузер покажет предупреждение — подтвердите переход.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())