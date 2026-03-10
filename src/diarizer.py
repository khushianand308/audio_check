import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

load_dotenv()

class Diarizer:
    """
    Local speaker diarization using Pyannote.audio.
    Lazy-loads the pipeline on the first call to diarize() for faster server startup.
    """
    def __init__(self, token=None):
        self.token = token or os.getenv("HF_TOKEN")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._loaded = False
        # Don't load at startup — wait until diarize() is called

    def _load_pipeline(self):
        """Loads the Pyannote pipeline if not already loaded (lazy init)."""
        if self._loaded:
            return
        self._loaded = True
        if self.token:
            try:
                print(f"Loading Pyannote diarization pipeline on {self.device}...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.token
                )
                if self.pipeline:
                    self.pipeline.to(self.device)
                    print("Pyannote pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading Pyannote pipeline: {e}")
                self.pipeline = None
        else:
            print("HF_TOKEN not found. Pyannote diarization will be skipped.")


    def diarize(self, audio_path):
        """
        Processes audio and returns list of speaker segments.
        Format: [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}, ...]
        """
        self._load_pipeline()  # Lazy-load on first call
        if not self.pipeline:
            return []

        try:
            print(f"Diarizing audio: {os.path.basename(audio_path)}")
            # Force exactly 2 speakers for call center dialogue
            diarization = self.pipeline(audio_path, num_speakers=2)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            print(f"Diarization complete. Found {len(segments)} segments.")
            return segments
        except Exception as e:
            print(f"Diarization error: {e}")
            return []

    def align_transcript(self, dg_words, py_segments):
        """
        Aligns Deepgram word-level timestamps with Pyannote speaker segments.
        dg_words: List of word dicts from Deepgram [{'word': 'hello', 'start': 0.1, 'end': 0.3}, ...]
        py_segments: List of speaker segments from Pyannote
        """
        if not dg_words or not py_segments:
            return []

        # Identify which speaker is the Agent based on high-probability keywords
        agent_speaker_id = None
        
        # Keywords suggesting an Agent introduction in Hindi/English
        agent_keywords = [
            "बोल", "रही", "हूं", "नमस्ते", "कॉल", "सेवा", "bank", "finance",
            "hdfc", "hdb", "financial", "services", "sakshi", "neha", "नेहा", "साक्षी"
        ]
        
        speaker_scores = {}
        for word_info in dg_words:
            word = word_info['word'].lower()
            mid = (word_info['start'] + word_info['end']) / 2
            for seg in py_segments:
                if seg['start'] <= mid <= seg['end']:
                    sid = seg['speaker']
                    if sid not in speaker_scores: speaker_scores[sid] = 0
                    if any(k in word for k in agent_keywords):
                        speaker_scores[sid] += 1
                    break
        
        if speaker_scores:
            agent_speaker_id = max(speaker_scores, key=speaker_scores.get)
            # If scores are zero or tied, pick the first speaker as default Agent
            if speaker_scores.get(agent_speaker_id, 0) == 0:
                agent_speaker_id = py_segments[0]['speaker'] if py_segments else None

        speaker_map = {}
        processed_transcript = []
        current_speaker = None
        current_text = []

        # Second pass: Assign roles
        for seg in py_segments:
            sid = seg['speaker']
            if sid not in speaker_map:
                if sid == agent_speaker_id:
                    speaker_map[sid] = "Agent"
                else:
                    speaker_map[sid] = "Customer"

        # Third pass: Process words with gap filling
        last_valid_role = "Agent"
        for word_info in dg_words:
            mid = (word_info['start'] + word_info['end']) / 2
            word_speaker = "UNKNOWN"
            for seg in py_segments:
                if seg['start'] <= mid <= seg['end']:
                    word_speaker = seg['speaker']
                    break
            
            role = speaker_map.get(word_speaker, last_valid_role)
            last_valid_role = role

            if role != current_speaker:
                if current_speaker and current_text:
                    processed_transcript.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = role
                current_text = [word_info['word']]
            else:
                current_text.append(word_info['word'])
        
        if current_speaker and current_text:
            processed_transcript.append(f"{current_speaker}: {' '.join(current_text)}")
            
        return "\n\n".join(processed_transcript)
            
        return "\n\n".join(processed_transcript)

if __name__ == "__main__":
    # Quick test
    d = Diarizer()
    if d.pipeline:
        res = d.diarize("uploads/clean_audio3.mp3")
        print(res[:5])
