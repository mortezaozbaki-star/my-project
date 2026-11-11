"""
Talking robot test (laptop): Russian or Kazakh. Brain = DeepSeek.

Chain: microphone -> speech-to-text -> DeepSeek -> text-to-speech -> speaker

Setup (Windows cmd, inside the venv):
    pip install SpeechRecognition pyaudio openai edge-tts pygame
    set DEEPSEEK_API_KEY=your-key-here

Run (pick the language):
    python robot_test.py ru        <- Russian
    python robot_test.py kk        <- Kazakh

Say "стоп" ("тоқта" in Kazakh) to quit. Ctrl+C also works.
Internet is needed for speech recognition, the voice and the AI.
"""

import asyncio
import os
import sys
import tempfile
import uuid

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import edge_tts
import pygame
import speech_recognition as sr
from openai import OpenAI, OpenAIError  # DeepSeek uses the OpenAI-compatible format

# ---------- Settings you can change ----------
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"   # check the model name in the official DeepSeek docs

LANGUAGES = {
    "ru": {
        "name": "Russian",
        "stt": "ru-RU",                       # speech recognition language
        "voice": "ru-RU-SvetlanaNeural",      # or "ru-RU-DmitryNeural"
        "hello": "Привет! Я готов.",
        "bye": "До свидания!",
        "sorry": "Извини, я не смог придумать ответ.",
        "stop_words": {"стоп", "stop", "выход"},
    },
    "kk": {
        "name": "Kazakh",
        "stt": "kk-KZ",
        "voice": "kk-KZ-AigulNeural",         # or "kk-KZ-DauletNeural"
        "hello": "Сәлем! Мен дайынмын.",
        "bye": "Сау болыңыз!",
        "sorry": "Кешіріңіз, жауап таба алмадым.",
        "stop_words": {"тоқта", "стоп", "stop"},
    },
}
# ---------------------------------------------

lang_key = sys.argv[1].lower() if len(sys.argv) > 1 else "ru"
if lang_key not in LANGUAGES:
    sys.exit("Usage: python robot_test.py ru   or   python robot_test.py kk")
L = LANGUAGES[lang_key]

SYSTEM_PROMPT = (
    "You are a friendly small robot talking with a student. "
    f"Always answer only in {L['name']}. "
    "Keep answers to one or two short sentences. "
    "Your answer will be read aloud, so do not use emojis, lists or markdown."
)

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    sys.exit("DEEPSEEK_API_KEY is not set. In cmd run: set DEEPSEEK_API_KEY=your-key")

client = OpenAI(api_key=api_key, base_url=BASE_URL)
recognizer = sr.Recognizer()
pygame.mixer.init()
history = []  # conversation memory (user / assistant turns)


def speak(text: str) -> None:
    """Mouth: print the text, make an mp3 with an online voice, play it."""
    print(f"Robot: {text}")
    path = os.path.join(tempfile.gettempdir(), f"robot_{uuid.uuid4().hex}.mp3")
    try:
        asyncio.run(edge_tts.Communicate(text, L["voice"]).save(path))
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.unload()
    except Exception as err:  # keep the robot alive even if the voice fails
        print(f"[voice error: {err}] (is the internet working?)")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def listen(mic: sr.Microphone):
    """Ears: record from the microphone and convert speech to text."""
    with mic as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None
    try:
        return recognizer.recognize_google(audio, language=L["stt"])
    except sr.UnknownValueError:
        return None  # could not understand
    except sr.RequestError as err:
        print(f"[speech service error: {err}] (is the internet working?)")
        return None


def think(user_text: str) -> str:
    """Brain: send the text to DeepSeek and return its answer."""
    history.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-9:]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=400,
        )
        answer = (response.choices[0].message.content or "").strip()
    except OpenAIError as err:
        history.pop()  # remove the message that failed
        print(f"[AI error: {err}]")
        return L["sorry"]
    if not answer:
        history.pop()
        return L["sorry"]
    history.append({"role": "assistant", "content": answer})
    return answer


def main() -> None:
    try:
        mic = sr.Microphone()
    except OSError as err:
        sys.exit(f"Microphone not found: {err}")

    with mic as source:
        print("Calibrating for background noise (1 second, stay quiet)...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

    speak(L["hello"])
    while True:
        heard = listen(mic)
        if not heard:
            continue
        print(f"You: {heard}")
        if heard.strip().lower() in L["stop_words"]:
            speak(L["bye"])
            break
        speak(think(heard))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
