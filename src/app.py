from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import torch
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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

@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Base dir = project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Setup directories with absolute paths
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
# Initialize analyzers (global singletons for speed)
audio_analyzer = AudioNoiseAnalyzer(target_sr=16000)
cleaner = AudioCleaner()
pro_transcriber = TranscriberPro()

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "gpu": torch.cuda.is_available()}

@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    manual_transcript: str = Form(None)
):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".mp3"
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    clean_path = os.path.join(PROCESSED_DIR, f"{file_id}_clean.wav")
    
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 1. Audio Audit (Original Signal)
        audio_res = audio_analyzer.analyze(raw_path)
        is_noisy = audio_res.get("is_noisy", False)
        
        # 2. Production Cleanup (Conditional) - Auto-clean noisy audio with DeepFilterNet
        target_path_for_asr = raw_path
        cleaned_file_url = None
        
        if is_noisy:
            print(f"Audio flagged as NOISY. Applying DeepFilterNet to: {file.filename}...")
            try:
                cleaner.clean_audio(raw_path, clean_path)
                if os.path.exists(clean_path):
                    target_path_for_asr = clean_path
                    cleaned_file_url = f"/data/processed/{file_id}_clean.wav"
                    print(f"DeepFilterNet cleanup complete. Clean file saved to: {clean_path}")
                else:
                    print(f"DeepFilterNet ran but no output file found. Using raw audio.")
            except Exception as clean_err:
                print(f"DeepFilterNet failed: {clean_err}. Using raw audio.")
        else:
            print(f"Audio is CLEAN. Bypassing DeepFilterNet for: {file.filename}.")
        
        # 3. Transcription Audit (Deepgram Pro)
        generated_transcript = None
        trans_res = None
        transcript_confidence = None
        transcribed_source = "raw"
        
        if pro_transcriber.client:
            # === CLEAN-FIRST STRATEGY ===
            # Primary: Use cleaned audio if available (better signal quality for ASR)
            # Fallback: Use raw audio if cleaned fails or doesn't exist
            
            if os.path.exists(clean_path):
                print(f"Transcribing with Deepgram Nova-2: {file.filename} (Primary: CLEANED audio)...")
                trans_res = pro_transcriber.transcribe(clean_path)
                transcribed_source = "cleaned"

                if "text" in trans_res:
                    generated_transcript = trans_res["text"]
                    transcript_confidence = trans_res.get("confidence")

                # Check if cleaned transcription failed (empty or very low confidence)
                is_failed_clean = (not generated_transcript or len(generated_transcript.strip()) < 5 or
                                   (transcript_confidence is not None and transcript_confidence < 0.2))

                if is_failed_clean:
                    print("Cleaned audio transcription failed or low-res. Falling back to RAW audio...")
                    trans_res = pro_transcriber.transcribe(raw_path)
                    transcribed_source = "raw"
                    if "text" in trans_res:
                        generated_transcript = trans_res["text"]
                        transcript_confidence = trans_res.get("confidence")
            else:
                # No cleaned audio available — transcribe raw directly
                print(f"Transcribing with Deepgram Nova-2: {file.filename} (RAW - no cleaned version)...")
                trans_res = pro_transcriber.transcribe(raw_path)
                transcribed_source = "raw"
                if "text" in trans_res:
                    generated_transcript = trans_res["text"]
                    transcript_confidence = trans_res.get("confidence")

            print(f"Transcription complete. Source: {transcribed_source}, Confidence: {transcript_confidence}")
        else:
            print("Deepgram API key not set. Skipping transcription.")

        # 4. Semantic Audit (Hallucination Guard)
        hallucination_flags = []
        if generated_transcript:
            hallucination_flags = pro_transcriber.hallucination_guard(generated_transcript)


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
            "manual_transcript": manual_transcript,
            "transcript_confidence": transcript_confidence,
            "transcribed_source": transcribed_source,
            "comparison": comparison,
            "hallucination_flags": hallucination_flags,
            "flags": reasons,
            "is_reliable": is_ready,  # True if Audio Math >= 0.75
            "reliability_score": readiness_score,  # Explicit Readiness Score 0-100%
            "raw_audio_url": raw_url,
            "clean_audio_url": cleaned_file_url
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # We keep the file for now, but in production we might delete it
        pass

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    
    # Fallback to index.html in root if static folder version is missing
    root_index = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(root_index):
        with open(root_index, "r") as f:
             return HTMLResponse(content=f.read())
    
    return HTMLResponse(content="Index not found. Please create static/index.html", status_code=404)

# Serve static files and data securely
app.mount("/data/processed", StaticFiles(directory=os.path.join(BASE_DIR, "data", "processed")), name="processed")
app.mount("/data/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "data", "uploads")), name="uploads")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8766, reload=False)
