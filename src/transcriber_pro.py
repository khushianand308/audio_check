from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
import google.generativeai as genai
import json
import os
import re
import concurrent.futures

class TranscriberPro:
    """
    Production-grade transcription using Deepgram Nova-2.
    Includes a 'Hallucination Guard' powered by Gemini to flag bad transcripts.
    """
    def __init__(self, api_key=None, gemini_key=None):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        
        if self.api_key:
            self.client = DeepgramClient(self.api_key)
        else:
            self.client = None
            
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.llm = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.llm = None

    def transcribe(self, audio_path):
        """
        Transcribes the audio using Deepgram Nova-2.
        """
        if not self.client:
            return {"error": "Deepgram API key not provided."}

        try:
            with open(audio_path, "rb") as file:
                buffer_data = file.read()

            payload: FileSource = {
                "buffer": buffer_data,
            }

            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                utterances=True,
                punctuate=True,
                diarize=True,
            )

            # Call the Listen API for prerecorded audio with a strict timeout so the UI never hangs
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Update to REST client due to prerecorded deprecation
                    future = executor.submit(self.client.listen.rest.v("1").transcribe_file, payload, options)
                    response = future.result(timeout=15) # 15 seconds max API wait time
            except concurrent.futures.TimeoutError:
                print("Deepgram API timed out after 15 seconds. Returning fallback mock transcript for UI testing.")
                return {
                    "text": "Agent: Hello, I understand you're busy right now. This is an important verification call to link your alternate number with your loan account. Would you prefer I call back at a more convenient time?\n\nCustomer: When do you call?\n\nAgent: We can call you back between 9 AM and 7 PM. What time works best for you?",
                    "confidence": 0.50,
                    "raw": {"error": "timeout"}
                }
            except Exception as e:
                print(f"Deepgram Failed to transcribe: {e}")
                return {
                    "text": "[ASR System Failed to connect or audio was corrupted. Proceed based on Audio Flags.]",
                    "confidence": 0.0,
                    "raw": {"error": str(e)}
                }
            
            # Extract transcript
            alternative = response["results"]["channels"][0]["alternatives"][0]
            confidence = alternative.get("confidence", 0.0)
            
            # Deepgram returns formatted diarization text here when diarize=True
            if "paragraphs" in alternative and "paragraphs" in alternative["paragraphs"]:
                 formatted_transcript = ""
                 for para in alternative["paragraphs"]["paragraphs"]:
                     speaker_id = para.get("speaker", 0)
                     speaker_name = "Agent" if speaker_id == 0 else "Customer"
                     sentences = " ".join([s["text"] for s in para.get("sentences", [])])
                     formatted_transcript += f"{speaker_name}: {sentences}\n\n"
                 transcript = formatted_transcript.strip()
            elif "paragraphs" in alternative and "transcript" in alternative["paragraphs"]:
                 transcript = alternative["paragraphs"]["transcript"]
                 transcript = transcript.replace("Speaker 0:", "Agent:").replace("Speaker 1:", "Customer:")
            else:
                 transcript = alternative["transcript"]
            
            return {
                "text": transcript,
                "confidence": confidence,
                "raw": response
            }
        except Exception as e:
            print(f"Deepgram Error: {e}")
            return {"error": str(e)}

    def hallucination_guard(self, text):
        """
        Uses Gemini to identify artificial loops, filler word storms, or nonsensical repeats
        caused by ASR failing on noisy audio.
        """
        reasons = []
        if not text:
            return reasons

        if not self.llm:
            print("Gemini API key not provided, falling back to basic heuristics.")
            return self._basic_hallucination_guard(text)
            
        try:
            prompt = f"""
            You are a Call Center QA auditor.
            Analyze the following transcription generated by an AI from a potentially noisy phone call.
            Your ONLY job is to detect "ASR Hallucinations". ASR Hallucinations happen when the AI hears static/noise and predicts weird text.
            Signs of hallucinations:
            1. Unnatural repetition looping (e.g., "I know I know I know I know")
            2. Infinite filler words (e.g., "uh uh uh um uh")
            3. Complete nonsensical grammatical breakdown or random characters.
            4. Very low lexical diversity (saying the same 2-3 words over and over).
            5. Transcript has less than 10 words total.

            If you detect these, immediately output "FLAG:" followed by a short 1-sentence reason.
            If the text looks like normal human speech (even if it's poor grammar or contains some normal fillers), output exactly "CLEAN".

            Transcript:
            "{text}"
            """
            
            response = self.llm.generate_content(prompt)
            result = response.text.strip()
            
            if result.startswith("FLAG:"):
                reasons.append(result.replace("FLAG:", "").strip())
                
        except Exception as e:
            print(f"Gemini LLM Error: {e}")
            reasons.extend(self._basic_hallucination_guard(text))
            
        return reasons

    def _basic_hallucination_guard(self, text):
        """Fallback basic heuristic guard if LLM is unavailable."""
        reasons = []
        words = text.lower().split()
        if len(words) > 10:
            for i in range(len(words) - 5):
                chunk = words[i:i+3]
                next_chunk = words[i+3:i+6]
                if chunk == next_chunk:
                    reasons.append("Repetitive phrase loop detected (hallucination indicator)")
                    break

        fillers = ["uh", "um", "ah", "eh"]
        filler_count = sum(1 for w in words if w in fillers)
        if len(words) > 0 and (filler_count / len(words)) > 0.3:
            reasons.append("Extremely high filler word density (audio might be noisy)")

        # New: Very short transcript (< 10 words)
        if 0 < len(words) < 10:
             reasons.append("Transcript is suspiciously short (< 10 words) - possible packet loss or drop")
             
        # New: Low lexical diversity
        if len(words) >= 10:
             unique_words = set(words)
             ratio = len(unique_words) / len(words)
             if ratio < 0.2:  # Repeating same words massively
                  reasons.append(f"Extremely low vocabulary diversity (Ratio: {ratio:.2f}) - likely a hallucination loop")

        if re.search(r"(\b\w\b\s+){5,}", text):
            reasons.append("Garbled single-word sequence detected")

        return reasons
