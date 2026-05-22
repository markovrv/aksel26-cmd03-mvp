"""
Voice Streaming Server (HTTPS/WSS) — распознавание речи через Whisper + синтез речи Silero
========================================================================================
Запустить: python fserver.py
Зависимости: pip install websockets faster-whisper numpy httpx torch silero omegaconf
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

import numpy as np
import torch

# Windows: добавляем пути к DLL библиотекам nvidia/cuda
if os.name == "nt":
    _venv_base = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
    _nv_paths = [
        os.path.join(_venv_base, "nvidia", "cublas", "bin"),
        os.path.join(_venv_base, "nvidia", "cudnn", "bin"),
        os.path.join(_venv_base, "nvidia", "cuda_nvrtc", "bin"),
    ]
    for _p in _nv_paths:
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
    _openssl_path = r"C:\Program Files\OpenSSL-Win64\bin"
    if os.path.isdir(_openssl_path):
        os.add_dll_directory(_openssl_path)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

# TTS конфигурация
TTS_MODEL_ID = "v4_ru"       # v4_ru | v3_1_ru | v5_ru | ...
TTS_SPEAKER = "xenia"        # xenia | aidar | baya | kseniya | eugeny
TTS_SAMPLE_RATE = 24000
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

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ---------- Сертификат ----------
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

def generate_self_signed_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        log.info("Сертификат уже существует.")
        return True
    log.info("Генерация самоподписанного сертификата...")
    try:
        subprocess.run(["openssl", "version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        log.error("OpenSSL не найден в системе.")
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
#  Whisper
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
            audio, language=LANGUAGE, beam_size=5,
            no_speech_threshold=NO_SPEECH_THRESHOLD, condition_on_previous_text=False,
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
        phrases = ["спасибо за внимание", "до свидания", "на этом всё", "конец записи"]
        if any(p in text_lower for p in phrases):
            return len(text_lower.split()) <= 5
        return False

# ====================================================================
#  TTS (Silero через пакет silero)
# ====================================================================
class LocalTTS:
    def __init__(self, speaker="xenia"):
        self.sample_rate = TTS_SAMPLE_RATE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"TTS использует устройство: {self.device}")

        log.info(f"Загрузка Silero TTS (модель: {TTS_MODEL_ID})...")
        import silero
        # Пробуем загрузить через silero-пакет
        try:
            self.model, _ = silero.silero_tts(language="ru", speaker=TTS_MODEL_ID)
        except RuntimeError as e:
            if "open file failed" in str(e):
                # PackageImporter не работает с кириллицей в пути
                # Меняем директорию и загружаем через относительный путь
                silero_dir = os.path.dirname(silero.__file__)
                model_rel_path = os.path.join("model", f"{TTS_MODEL_ID}.pt")
                old_cwd = os.getcwd()
                os.chdir(silero_dir)
                try:
                    imp = torch.package.PackageImporter(model_rel_path)
                    self.model = imp.load_pickle("tts_models", "model")
                finally:
                    os.chdir(old_cwd)
            else:
                raise
        self.model.to(self.device)
        self._speaker = speaker
        log.info(f"Silero TTS загружен (голос: {self._speaker})")

    def synthesize(self, text: str) -> bytes:
        if not text or len(text.strip()) < 1:
            return b""
        try:
            audio = self.model.apply_tts(
                text=text, speaker=self._speaker,
                sample_rate=self.sample_rate, put_accent=True, put_yo=True,
            )
            audio = audio.numpy()
            if len(audio) > 0:
                mx = np.abs(audio).max()
                if mx > 0:
                    audio = audio / mx * 0.95
                audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
            pcm16 = (audio * 32767).astype(np.int16).tobytes()
            return pcm16
        except Exception as e:
            log.error(f"Ошибка TTS: {e}")
            return b""

# ====================================================================
#  Инициализация
# ====================================================================
recognizer = WhisperRecognizer()
tts = LocalTTS(speaker=TTS_SPEAKER)

def transcribe_audio(pcm_bytes: bytes) -> str:
    return recognizer.transcribe(pcm_bytes)

def rms(pcm_bytes: bytes) -> float:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

# ---------- Транскрипты ----------
def list_transcript_files() -> list:
    files = []
    if not os.path.isdir(TRANSCRIPTS_DIR):
        return files
    for fname in sorted(os.listdir(TRANSCRIPTS_DIR), reverse=True):
        fpath = os.path.join(TRANSCRIPTS_DIR, fname)
        if os.path.isfile(fpath) and fname.endswith(".txt"):
            stat = os.stat(fpath)
            files.append({"name": fname, "size": stat.st_size, "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()})
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    fname = f"{prefix}_{ts}_{uid}.txt"
    fpath = os.path.join(TRANSCRIPTS_DIR, fname)
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        log.info(f"Стенограмма сохранена: {fpath}")
        return fname
    except Exception as e:
        log.error(f"Ошибка сохранения стенограммы: {e}")
        return None

# ---------- LLM ----------
def call_llm_sync(messages: list) -> dict:
    if not HAS_HTTPX:
        return {"error": "Библиотека httpx не установлена."}
    if not LLM_API_KEY:
        return {"error": "LLM_API_KEY не настроен."}
    url = f"{LLM_API_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.7}
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"response": content}
    except httpx.HTTPStatusError as e:
        log.error(f"LLM HTTP ошибка: {e.response.status_code}")
        return {"error": f"LLM вернул ошибку {e.response.status_code}"}
    except httpx.RequestError as e:
        log.error(f"LLM ошибка соединения: {e}")
        return {"error": f"Не удалось подключиться к LLM: {e}"}
    except Exception as e:
        log.error(f"LLM ошибка: {e}")
        return {"error": str(e)}

# ---------- HTTP ----------
HTML_PATH = os.path.join(os.path.dirname(__file__), "voice-recorder.html")

def make_json_response(data: dict, status: int = 200) -> Response:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = Headers([("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body))), ("Cache-Control", "no-cache"), ("Access-Control-Allow-Origin", "*")])
    return Response(status, "OK" if status == 200 else "Error", headers, body)

def make_html_response(body: bytes, status: int = 200) -> Response:
    headers = Headers([("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body))), ("Cache-Control", "no-cache")])
    return Response(status, "OK" if status == 200 else "Error", headers, body)

def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    path = request.path.rstrip("/")
    log.info(f"HTTP {path}")
    if path == "/api/chat":
        try:
            data = json.loads(request.body or b"")
            messages = data.get("messages", [])
            if not messages:
                return make_json_response({"error": "Поле 'messages' обязательно"}, 400)
            result = call_llm_sync(messages)
            return make_json_response(result, 500 if "error" in result else 200)
        except json.JSONDecodeError:
            return make_json_response({"error": "Некорректный JSON"}, 400)
        except Exception as e:
            log.error(f"Ошибка /api/chat: {e}")
            return make_json_response({"error": str(e)}, 500)
    if path == "/api/health":
        return make_json_response({"status": "ok", "llm_configured": bool(LLM_API_KEY)})
    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            html_content = f.read().replace("ws://", "wss://")
        body = html_content.encode("utf-8")
    except FileNotFoundError:
        body = b"<h1>Voice Stream</h1><p>404 HTML not found</p>"
    except Exception as e:
        log.error(f"Ошибка чтения HTML: {e}")
        body = b"<h1>Voice Stream</h1><p>500 Error</p>"
    return make_html_response(body)

# ---------- WebSocket ----------
class ClientSession:
    def __init__(self):
        self.buffer = bytearray()
        self.silence_count = 0
        self.current_text = ""
        self.session_id = uuid.uuid4().hex[:12]
        self.reconnected = False
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_name = f"live_{ts}_{self.session_id}.txt"
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
                ts = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{ts}] {text}\n")
            log.info(f"Стенограмма дописана: {self.file_name}")
        except Exception as e:
            log.error(f"Ошибка записи стенограммы: {e}")

async def handle_client(ws):
    addr = ws.remote_address
    session = ClientSession()

    await ws.send(json.dumps({"type": "session", "session_id": session.session_id}))
    await ws.send(json.dumps({"type": "engine", "engine": "whisper"}))
    await ws.send(json.dumps({"type": "tts_ready", "speaker": TTS_SPEAKER, "sample_rate": TTS_SAMPLE_RATE}))

    try:
        async for message in ws:
            if isinstance(message, bytes):
                session.buffer.extend(message)
                level = rms(message)
                is_silent = level < SILENCE_THRESHOLD
                session.silence_count = session.silence_count + 1 if is_silent else 0
                await ws.send(json.dumps({"type": "level", "rms": round(level)}))

                if len(session.buffer) >= CHUNK_BYTES * 3 and session.silence_count >= SILENCE_CHUNKS:
                    audio_snapshot = bytes(session.buffer)
                    session.buffer.clear()
                    session.silence_count = 0
                    await ws.send(json.dumps({"type": "status", "text": "Распознаю..."}))
                    try:
                        text = await asyncio.get_event_loop().run_in_executor(None, transcribe_audio, audio_snapshot)
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
                                text = await asyncio.get_event_loop().run_in_executor(None, transcribe_audio, audio_snapshot)
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
                        await ws.send(json.dumps({"type": "file_saved" if fname else "error", "filename": fname or "", "text": "Стенограмма сохранена" if fname else "Нет текста для сохранения"}))

                    elif action == "list_transcripts":
                        await ws.send(json.dumps({"type": "transcript_list", "files": list_transcript_files()}))

                    elif action == "get_transcript":
                        content = read_transcript_file(cmd.get("filename", ""))
                        if content is not None:
                            await ws.send(json.dumps({"type": "transcript_content", "filename": cmd["filename"], "content": content}))
                        else:
                            await ws.send(json.dumps({"type": "error", "text": f"Файл '{cmd.get('filename')}' не найден"}))

                    elif action == "delete_transcript":
                        ok = delete_transcript_file(cmd.get("filename", ""))
                        await ws.send(json.dumps({"type": "file_deleted" if ok else "error", "filename": cmd.get("filename")}))

                    elif action == "reconnect":
                        session.reconnected = True
                        log.info(f"Клиент {addr} переподключился (сессия {session.session_id})")
                        await ws.send(json.dumps({"type": "reconnect_ack", "session_id": session.session_id}))

                    elif action == "chat":
                        messages = cmd.get("messages", [])
                        if not messages:
                            await ws.send(json.dumps({"type": "error", "text": "Поле 'messages' обязательно"}))
                        else:
                            try:
                                result = await asyncio.get_event_loop().run_in_executor(None, call_llm_sync, messages)
                                await ws.send(json.dumps({"type": "chat_response", **result}))
                            except Exception as e:
                                await ws.send(json.dumps({"type": "chat_response", "error": str(e)}))

                    elif action == "tts":
                        text = cmd.get("text", "")
                        speaker = cmd.get("speaker", tts._speaker)
                        if not text.strip():
                            await ws.send(json.dumps({"type": "error", "text": "Поле 'text' пусто"}))
                        else:
                            log.info(f"TTS: '{text[:50]}...' (голос: {speaker})")
                            await ws.send(json.dumps({"type": "tts_status", "text": "Синтезирую..."}))
                            try:
                                old = tts._speaker
                                if speaker != old:
                                    tts._speaker = speaker
                                pcm_bytes = await asyncio.get_event_loop().run_in_executor(None, tts.synthesize, text)
                                if speaker != old:
                                    tts._speaker = old
                                if pcm_bytes:
                                    await ws.send(json.dumps({"type": "tts_start", "sample_rate": TTS_SAMPLE_RATE, "channels": 1, "format": "pcm16", "length": len(pcm_bytes)}))
                                    await ws.send(pcm_bytes)
                                    await ws.send(json.dumps({"type": "tts_end"}))
                                    log.info(f"TTS отправлено {len(pcm_bytes)} байт")
                                else:
                                    await ws.send(json.dumps({"type": "error", "text": "TTS не выдал аудио"}))
                            except Exception as e:
                                log.error(f"Ошибка TTS: {e}")
                                await ws.send(json.dumps({"type": "error", "text": f"Ошибка TTS: {e}"}))

                except json.JSONDecodeError:
                    pass

    except websockets.exceptions.ConnectionClosed as e:
        log.info(f"[Whisper] Клиент отключился: {addr} (код {e.code}) (сессия {session.session_id})")
    except Exception as e:
        log.error(f"[Whisper] Ошибка соединения {addr}: {e}", exc_info=True)

# ---------- Запуск ----------
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
    async with websockets.serve(handle_client, HOST, PORT, process_request=process_request, ssl=ssl_context):
        log.info(f"WebSocket-сервер (WSS) запущен на wss://{local_ip}:{PORT}")
        log.info(f"HTML-интерфейс: https://{local_ip}:{PORT}")
        log.info(f"Стенограммы: '{TRANSCRIPTS_DIR}/'")
        log.info(f"Движок: Whisper ({MODEL_SIZE}), TTS: Silero (голос: {TTS_SPEAKER})")
        if LLM_API_KEY:
            log.info(f"LLM API: {LLM_API_URL}, модель={LLM_MODEL}")
        else:
            log.info("LLM API не настроен.")
        log.info("Ожидаю подключений...")
        log.warning("Самоподписанный сертификат — подтвердите переход в браузере.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())