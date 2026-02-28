from faster_whisper import WhisperModel
import os

class WhisperTranscriber:
    """
    Handles audio transcription using Faster-Whisper.
    Uses the 'tiny' model by default for speed on CPU.
    """
    def __init__(self, model_size="tiny", device="cpu", compute_type="int8"):
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Note: The model will be downloaded to ~/.cache/huggingface/hub by default
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path):
        """Transcribes the audio and returns the text."""
        try:
            self._load_model()
            
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            segments = list(segments)
            full_text = " ".join([seg.text.strip() for seg in segments])
            
            return {
                "text": full_text,
                "language": info.language,
                "probability": info.language_probability,
                "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            }
        except Exception as e:
            print(f"Faster-Whisper Error: {e}")
            return None
