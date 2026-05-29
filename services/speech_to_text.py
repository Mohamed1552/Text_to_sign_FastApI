import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment


class SpeechToTextSR:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    # =========================
    # 1) من WAV مباشر
    # =========================
    def transcribe_file(self, audio_path):
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)

        try:
            text = self.recognizer.recognize_google(audio, language="ar-EG")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise Exception(f"API error: {e}")

    # =========================
    # 2) من MP3 أو أي صوت (تحويل تلقائي)
    # =========================
    def transcribe_any(self, audio_path):
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            tmp_path = tmp.name

        try:
            text = self.transcribe_file(tmp_path)
        finally:
            os.remove(tmp_path)

        return text

    # =========================
    # 3) من المايك مباشرة
    # =========================
    def record_and_transcribe(self, duration=5):
        with sr.Microphone() as source:
            print("🎤 اتكلم دلوقتي...")
            audio = self.recognizer.record(source, duration=duration)

        try:
            text = self.recognizer.recognize_google(audio, language="ar-EG")
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise Exception(f"API error: {e}")