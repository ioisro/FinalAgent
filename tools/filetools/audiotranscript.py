import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AudioTranscriber:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Uses OPENAI_API_KEY from environment

    def __call__(self, audio_path_or_file) -> str:
        """
        Transcribes the given audio file using OpenAI's gpt-4o-transcribe model and returns the transcript as text.
        Accepts either a file path (str) or a file-like object (e.g., BytesIO).
        """
        try:
            if isinstance(audio_path_or_file, str):
                audio_file = open(audio_path_or_file, "rb")
                close_after = True
            else:
                audio_file = audio_path_or_file
                close_after = False

            transcription = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
            if close_after:
                audio_file.close()
            return transcription.text
        except Exception as e:
            return f"Transcription failed: {e}"