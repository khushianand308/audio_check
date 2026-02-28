from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import uuid
from dotenv import load_dotenv

from audio_cleaner import AudioCleaner
from transcriber_pro import TranscriberPro
from noise_analyzer import AudioNoiseAnalyzer
from fuzzywuzzy import fuzz

load_dotenv()

app = FastAPI(title="AudioSense Pro API")

# Setup directories
UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
# Initialize analyzers (lazy load if possible or global)
audio_analyzer = AudioNoiseAnalyzer(target_sr=16000)
cleaner = AudioCleaner()

@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    manual_transcript: str = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".mp3"
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    clean_path = os.path.join(PROCESSED_DIR, f"{file_id}_clean{ext}")
    
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Audio Audit (Original Signal)
        audio_res = audio_analyzer.analyze(raw_path)
        is_noisy = audio_res.get("is_noisy", False)
        
        # 2. Production Cleanup (Conditional)
        target_path_for_asr = raw_path
        cleaned_file_url = None
        
        # [STEP 1 FOCUS] Bypassing Heavy Models: DeepFilterNet temporarily disabled for instant audio scoring tests
        # if is_noisy:
        #     print(f"Audio flagged as NOISY. Applying DeepFilterNet to: {file.filename}...")
        #     cleaner.clean_audio(raw_path, clean_path)
        #     target_path_for_asr = clean_path
        #     cleaned_file_url = f"/data/processed/clean_{file.filename}"
        # else:
        #     print(f"Audio is CLEAN. Bypassing DeepFilterNet for: {file.filename}...")
        
        # 3. Transcription Audit (Deepgram Pro)
        generated_transcript = None
        trans_res = None
        
        # [STEP 1 FOCUS] Bypassing Heavy Models: Deepgram temporarily disabled for instant audio scoring tests
        # pro_transcriber = TranscriberPro()
        # if pro_transcriber.client:
        #     print("Transcribing with Deepgram Pro...")
        #     trans_res = pro_transcriber.transcribe(target_path_for_asr) 
        #     if "text" in trans_res:
        #         generated_transcript = trans_res["text"]

        # 4. Semantic Audit (Hallucination Guard)
        hallucination_flags = []
        # if generated_transcript:
        #     hallucination_flags = pro_transcriber.hallucination_guard(generated_transcript)

        # 5. Comparison Logic
        comparison = None
        if generated_transcript and manual_transcript:
            similarity = fuzz.ratio(generated_transcript.lower(), manual_transcript.lower())
            comparison = {
                "similarity": similarity,
                "is_match": similarity > 85,
                "diff_score": 100 - similarity
            }

        # 6. Unified Flagging
        is_reliable = not audio_res.get("is_noisy", False)
        reasons = audio_res.get("reasons", [])
        
        if hallucination_flags:
            is_reliable = False
            reasons.extend(hallucination_flags)
            
        # 7. Final Output Construction
        readiness_score = audio_res.get("transcript_readiness", 0)
        is_ready = audio_res.get("is_good_for_transcript", False)
        
        # Override the flag if semantic errors exist
        if hallucination_flags:
            is_ready = False
            reasons.extend(hallucination_flags)

        raw_url = f"/data/uploads/{file_id}{ext}"
        
        return {
            "status": "success",
            "audio_audit": audio_res,
            "transcript": generated_transcript,
            "comparison": comparison,
            "flags": reasons,
            "is_reliable": is_ready,  # True if Audio Math >= 0.75
            "reliability_score": readiness_score,  # Explicit Readiness Score 0-100%
            "raw_file_url": raw_url,
            "cleaned_file_url": cleaned_file_url
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # We keep the file for now, but in production we might delete it
        pass

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r") as f:
            return f.read()
    elif os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return f.read()
    return "Index not found. Please create index.html"

# Serve static files and data securely
app.mount("/data/processed", StaticFiles(directory="data/processed"), name="processed")
app.mount("/data/uploads", StaticFiles(directory="data/uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
