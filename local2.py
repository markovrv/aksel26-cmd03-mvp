import json
import os
import asyncio
import threading
import time
import sys
import queue
import re
import requests
from typing import Optional
from collections import deque

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==== КОНФИГУРАЦИЯ ====
# Global history of last 5 interactions with LLM
recent_history = deque(maxlen=5)
SAMPLE_RATE = 16000  # Vosk требует 16kHz
OUT_RATE = 24000  # Частота для синтезатора Silero (24kHz по умолчанию)
CHANNELS = 1
FRAME_MS = 20
IN_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)  # 320 samples

# Пути к моделям Vosk
VOSK_MODEL_PATH = "vosk-model-ru-0.42"

# LLM конфигурация
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "saiga_yandexgpt_8b_gguf"


# ==== ИНИЦИАЛИЗАЦИЯ ЛОКАЛЬНЫХ МОДЕЛЕЙ ====
class LocalSTT:
    """Локальное распознавание речи через Vosk"""

    def __init__(self, model_path=VOSK_MODEL_PATH):
        print("🎙️ Загрузка Vosk модели...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.partial_text = ""
        self.final_text = ""
        print(f"✓ Vosk загружен из {model_path}")

    def process_audio(self, audio_bytes: bytes) -> tuple:
        if self.recognizer.AcceptWaveform(audio_bytes):
            result = json.loads(self.recognizer.Result())
            self.final_text = result.get("text", "")
            self.partial_text = ""
            return self.final_text, ""
        else:
            partial = json.loads(self.recognizer.PartialResult())
            self.partial_text = partial.get("partial", "")
            return "", self.partial_text

    def reset(self):
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.partial_text = ""
        self.final_text = ""


class LocalTTS:
    """
    Локальный синтез речи через Silero с параллельной генерацией
    """

    AVAILABLE_VOICES = {
        "xenia": "женский, спокойный",
        "aidar": "мужской, уверенный",
        "baya": "женский, эмоциональный",
        "kseniya": "женский, деловой стиль",
        "eugeny": "мужской, высокое качество",
    }

    MODEL_URLS = {
        "v4_ru": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
        "v3_ru": "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
    }

    def __init__(self, speaker="kseniya", model_version="v4_ru", model_filename=None):
        self.speaker = speaker
        self.sample_rate = 24000
        self.device = torch.device("cpu")
        
        self._stop_synthesis = False
        self._is_synthesizing = False
        
        if speaker not in self.AVAILABLE_VOICES:
            print(f"⚠️ Голос '{speaker}' не найден, используем 'xenia'")
            self.speaker = "xenia"

        if model_filename is None:
            model_filename = f"silero_{model_version}.pt"

        self.model_path = model_filename
        self._ensure_model_downloaded(model_version)

        print(f"🔊 Загрузка Silero TTS из {self.model_path}...")
        try:
            self.model = torch.package.PackageImporter(self.model_path).load_pickle(
                "tts_models", "model"
            )
            self.model.to(self.device)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели Silero: {e}")
            raise
        print(f"✓ Silero загружен (голос: {self.speaker})")

    def _ensure_model_downloaded(self, model_version):
        if os.path.exists(self.model_path):
            print(f"✓ Модель найдена: {self.model_path}")
            return

        url = self.MODEL_URLS.get(model_version, self.MODEL_URLS["v4_ru"])
        print(f"📥 Скачивание модели Silero ({model_version})...")
        print(f"   Источник: {url}")

        try:
            torch.hub.download_url_to_file(url, self.model_path)
            print(f"✓ Модель загружена")
        except Exception as e:
            print(f"✗ Ошибка скачивания: {e}")
            sys.exit(1)

    def stop(self):
        self._stop_synthesis = True

    def reset_stop_flag(self):
        self._stop_synthesis = False

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)

    def _apply_fade(self, audio: np.ndarray, fade_ms: int = 5) -> np.ndarray:
        if len(audio) < 100:
            return audio

        fade_samples = int(self.sample_rate * fade_ms / 1000)
        fade_samples = min(fade_samples, len(audio) // 10)

        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out

        return audio

    def synthesize(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) < 1:
            return np.array([], dtype=np.float32)

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                put_accent=True,
                put_yo=True,
            )
            audio = audio.numpy()
            audio = self._normalize_audio(audio)
            audio = self._apply_fade(audio, fade_ms=5)
            return audio
        except Exception as e:
            print(f"\n⚠️ Ошибка синтеза: {e}")
            return np.array([], dtype=np.float32)


# ==== КЛАСС ДЛЯ ВЫВОДА АУДИО ====
class AudioOut:
    """Управляет выходным аудиопотоком"""

    def __init__(self, rate: int = OUT_RATE):
        self.queue = queue.Queue(maxsize=200)
        self.closed = False
        self.stream = sd.RawOutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=4096,
            latency="high",
            callback=self._audio_callback,
        )
        self.stream.start()

    def _audio_callback(self, outdata, frames, time_info, status):
        try:
            chunk = self.queue.get_nowait()
        except Exception:
            outdata[:] = b"\x00" * (frames * 2)
            return

        need = frames * 2
        if len(chunk) >= need:
            outdata[:] = chunk[:need]
            rest = chunk[need:]
            if rest:
                try:
                    self.queue.queue.appendleft(rest)
                except Exception:
                    pass
        else:
            outdata[: len(chunk)] = chunk
            outdata[len(chunk) :] = b"\x00" * (need - len(chunk))

    def write(self, pcm16_bytes: bytes):
        if self.closed or not pcm16_bytes:
            return
        try:
            self.queue.put_nowait(pcm16_bytes)
        except Exception:
            self.clear()

    def write_float32(self, audio: np.ndarray):
        if self.closed or len(audio) == 0:
            return
        pcm16 = (audio * 32767).astype(np.int16).tobytes()
        self.write(pcm16)

    def clear(self):
        try:
            with self.queue.mutex:
                self.queue.queue.clear()
        except Exception:
            pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# ==== КЛАСС ДЛЯ ЗАПИСИ С МИКРОФОНА ====
class MicStreamer:
    """Передает аудио с микрофона в асинхронную очередь"""

    def __init__(self, output_queue: asyncio.Queue, can_stream_event: asyncio.Event):
        self.output_queue = output_queue
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.can_stream_event = can_stream_event

    def start(self):
        self.loop = asyncio.get_running_loop()
        self.running = True
        threading.Thread(target=self._streaming_loop, daemon=True).start()

    def float_to_pcm16(self, data: np.ndarray) -> bytes:
        data = np.clip(data, -1.0, 1.0)
        return (data * 32767).astype(np.int16).tobytes()

    def _streaming_loop(self):
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=IN_SAMPLES,
                latency="low",
            ) as stream:
                while self.running:
                    if not self.can_stream_event.is_set():
                        time.sleep(0.01)
                        continue

                    data, _ = stream.read(IN_SAMPLES)
                    pcm = self.float_to_pcm16(data.reshape(-1))

                    if self.loop and not self.loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self.output_queue.put(pcm), self.loop
                        )
                        try:
                            future.result(timeout=0.2)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[MIC ERROR] {e}")

    def stop(self):
        self.running = False


# ==== ОСНОВНОЙ КЛАСС ГОЛОСОВОГО АГЕНТА ====
class VoiceAgent:
    def _log(self, entry: str):
        """Append a line to log.txt in the script directory."""
        try:
            log_path = os.path.join(os.path.dirname(__file__), "log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    def __init__(self, debug=False):
        self.debug = debug
        self.tts = LocalTTS(speaker="kseniya")
        self.audio_out = AudioOut()
        self.running = False
        self.response_epoch = 0
        self.current_response_epoch = None
        self.current_response_task = None
        
        # Тестируем подключение к LLM
        self._test_connections()
        
        print("🎤 Загрузка Vosk модели...")
        self.stt = LocalSTT()

        # Озвучиваем стартовое сообщение
        startup_msg = "Голосовой ассистент запущен! Говорите в микрофон. Нажмите Ctrl+C для выхода."
        audio = self.tts.synthesize(startup_msg)
        if audio.size:
            self.audio_out.write_float32(audio)

    def _test_connections(self):
        """Тестирует подключение к LM Studio и TTS"""
        API_URL = LM_STUDIO_URL
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты вежливый русскоязычный ассистент.",
                },
                {
                    "role": "user",
                    "content": "Привет!",
                },
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": True,
        }

        print("🔗 Проверка подключения к LM Studio...")

        try:
            with requests.post(
                API_URL, headers=headers, json=payload, timeout=10, stream=True
            ) as response:
                response.raise_for_status()
                print("✅ LM Studio доступен и работает")
                response.close()

            print("🔊 Тестирование синтезатора...")
            test_audio = self.tts.synthesize("Голосовой ассистент успешно запущен!")
            if test_audio.size:
                print("✅ Синтезатор работает")
            else:
                print("❌ Ошибка синтеза")

        except requests.exceptions.ConnectionError:
            print("❌ Не удалось подключиться к LM Studio. Проверьте, что сервер запущен.")
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")

    def _is_sentence_end(self, text: str) -> bool:
        """Проверяет, закончилось ли предложение"""
        if not text:
            return False
        end_chars = '.!?;:'
        return text[-1] in end_chars

    def _synthesize_and_play(self, phrase: str, epoch: int):
        """Синтезирует и воспроизводит фразу в отдельном потоке"""
        try:
            if epoch != self.response_epoch:
                return
            
            print(f"🔊 Озвучивание: {phrase}")
            audio = self.tts.synthesize(phrase)
            
            if epoch != self.response_epoch:
                return
            
            if len(audio) > 0:
                self.audio_out.write_float32(audio)
                
        except Exception as e:
            print(f"⚠️ Ошибка воспроизведения: {e}")

    async def process_response_streaming(self, text: str):
        """
        Потоковая обработка ответа LLM с немедленным воспроизведением
        """
        my_epoch = self.response_epoch
        self.current_response_epoch = my_epoch
        
        print(f"🤖 Запрос к LLM: {text}")
        
        API_URL = LM_STUDIO_URL
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты вежливый русскоязычный ассистент. Отвечай кратко и по делу. Не используй форматирование.",
                },
                # recent interaction history (up to 5 exchanges)
                *list(recent_history),
                {"role": "user", "content": text},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
            "stream": True,
        }

        try:
            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=60, stream=True
            )
            response.raise_for_status()
            
            accumulated_text = ""
            current_phrase = ""
            
            # Используем ThreadPoolExecutor для параллельного синтеза
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=2)
            
            # Читаем поток LLM
            for line in response.iter_lines():
                if not line:
                    continue
                    
                if my_epoch != self.response_epoch:
                    print("⏹️ Ответ прерван пользователем")
                    break
                
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        
                        if content:
                            accumulated_text += content
                            current_phrase += content
                            
                            # Показываем накопленный текст
                            # print(f"\r💬 {accumulated_text}", end="", flush=True)
                            
                            # Проверяем, закончилась ли фраза
                            if self._is_sentence_end(current_phrase):
                                phrase_to_play = current_phrase.strip()
                                current_phrase = ""
                                
                                # Запускаем синтез в отдельном потоке
                                if phrase_to_play:
                                    executor.submit(
                                        self._synthesize_and_play, 
                                        phrase_to_play, 
                                        my_epoch
                                    )
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"⚠️ Ошибка парсинга: {e}")
            
            # Отправляем остаток текста, если есть
            if current_phrase.strip() and my_epoch == self.response_epoch:
                executor.submit(
                    self._synthesize_and_play, 
                    current_phrase.strip(), 
                    my_epoch
                )
            
            # Даем время на завершение последнего синтеза
            await asyncio.sleep(0.1)
            
            # print(f"\n🤖 Полный ответ: {accumulated_text}")
            await asyncio.sleep(0.3)
            
            executor.shutdown(wait=False)
            # Save recent interaction (user query and assistant answer) to history (max 5 exchanges)
            recent_history.append({"role": "user", "content": text})
            recent_history.append({"role": "assistant", "content": accumulated_text})
            
        except requests.exceptions.ConnectionError:
            error_msg = "Не удалось подключиться к LM Studio. Проверьте, что сервер запущен."
            print(f"❌ {error_msg}")
            audio = self.tts.synthesize(error_msg)
            if len(audio) > 0:
                self.audio_out.write_float32(audio)
        except Exception as e:
            error_msg = f"Ошибка: {e}"
            print(f"❌ {error_msg}")
        finally:
            response.close()

    async def stop_current_response(self):
        """Принудительно останавливает текущий ответ"""
        if self.current_response_task and not self.current_response_task.done():
            self.current_response_task.cancel()
            try:
                await self.current_response_task
            except asyncio.CancelledError:
                pass
        
        self.response_epoch += 1
        self.current_response_epoch = None
        self.tts.stop()
        self.audio_out.clear()
        await asyncio.sleep(0.1)

    async def run(self):
        """Запуск голосового агента"""
        print("\n🎙️ Голосовой ассистент запущен!")
        print("Говорите в микрофон. Нажмите Ctrl+C для выхода.")
        print("=" * 50)

        mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        can_stream_input = asyncio.Event()
        can_stream_input.set()

        mic = MicStreamer(mic_queue, can_stream_input)
        mic.start()

        audio_buffer = b""
        self.running = True

        try:
            while self.running:
                try:
                    pcm = await asyncio.wait_for(mic_queue.get(), timeout=0.1)
                    audio_buffer += pcm

                    if len(audio_buffer) >= SAMPLE_RATE // 3:
                        final, partial = self.stt.process_audio(audio_buffer)
                        audio_buffer = b""

                        if final and len(final.strip()) > 0:
                            print(f"\n📝 Распознано: {final}")
                            await self.stop_current_response()
                            self.current_response_task = asyncio.create_task(
                                self.process_response_streaming(final)
                            )
                            self.stt.reset()

                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    break
                
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n👋 Завершение работы...")
        except Exception as e:
            print(f"\n❌ Непредвиденная ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            await self.stop_current_response()
            mic.stop()
            self.audio_out.close()
            await asyncio.sleep(0.5)

# ==== ЗАПУСК ====
async def main():
    agent = VoiceAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())