# AudioSense Pro 🎙️
**Neural Call Audit System** — AI-powered audio quality analysis for call center recordings.

Automatically scores, cleans, transcribes, and diarizes phone call recordings using a 7-step GPU neural pipeline.

---

## ⚡ Quick Start

```bash
# Start
cd /home/ubuntu/bg_noise_check
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8766
```

```bash
# Stop
pkill -f uvicorn

# Force stop (if above doesn't work)
pkill -f uvicorn -9
```

Open UI at: **http://localhost:8766**

---

## What It Does

Upload any `.mp3` or `.wav` call recording. The system will:

1. **Score the audio quality** (0–100% Transcript Readiness Score)
2. **Flag specific problems** — background noise, overlapping speakers, mic clipping, dead air
3. **Automatically clean noisy audio** using DeepFilterNet (AI noise suppressor)
4. **Transcribe** using Deepgram Nova-2 (Clean audio first, Raw as fallback)
5. **Diarize** speakers with Pyannote + Gemini (Agent vs Customer labels)
6. **Compare** the original and enhanced audio side-by-side in the browser
7. **Log a session history** of every file audited

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Server** | FastAPI + Uvicorn |
| **Audio Math** | NumPy, SciPy (SNR, Spectral Flux, Kurtosis, VAD) |
| **AI Quality Score** | Microsoft DNSMOS via `speechmos` + ONNX Runtime |
| **AI Noise Cleaner** | DeepFilterNet3 (GPU-accelerated, lossless WAV output) |
| **Transcription** | Deepgram Nova-2 (Clean-first, Raw fallback) |
| **Diarization** | Pyannote 3.1 (Acoustic) + Gemini 2.0 (Linguistic) |
| **Frontend** | Vanilla HTML, CSS, JavaScript |

---

## The 5 Audio Health Checks

| Check | What It Detects |
|---|---|
| **SNR** | Background noise level (hum, hiss) |
| **Speech Density** | Too much silence / dead air |
| **Clipping** | Mic distortion from too-loud recording |
| **Spectral Flux** | Two people talking over each other |
| **Kurtosis** | Chaotic energy spikes (overlap indicator) |

Combined with the **Microsoft DNSMOS AI score** (1–5), these produce the final **Transcript Readiness Score**.

---

## How It Works (Plain English)

### The 5 Math Checks (Acoustic Heuristics)

#### 1. SNR (Signal-to-Noise Ratio)
**What it means:** How loud is the person speaking compared to the background noise?

> Imagine trying to talk to a friend at a loud party. Your friend's voice is the "Signal," and the party music is the "Noise." SNR measures if the voice is loud and clear enough to be understood over the background.

#### 2. Speech Density (VAD)
**What it means:** How much of the audio is actual talking versus dead silence?

> Like a bad radio show with 40 seconds of dead air for every 10 seconds of talking. This ensures the file is an actual phone call and not mostly someone sitting on mute.

#### 3. Clipping Detection
**What it means:** Is the microphone "blown out" and distorted?

> Have you ever yelled too closely into a cheap microphone and the recording sounded fuzzy and harsh? That happens when the sound is literally too big for the mic to handle. This checks for that exact distortion.

#### 4 & 5. Spectral Flux & Kurtosis (Overlap Detection)
**What it means:** Are two people constantly talking over each other?

> Think of a normal conversation like a smooth dance — one person speaks, then the other. Two people shouting simultaneously makes the sound waves crash and become chaotic. These tools detect that "chaos" to warn that overlapping chatter is happening.

---

### The Artificial Intelligence Layer

Math alone is sometimes rigid. A file might pass the math test but still sound robotic. That's why AI is layered on top.

#### 1. Microsoft DNSMOS (The Robot Judge)
**What it means:** An AI that grades the call exactly like a human would, from 1 to 5 stars.

> Microsoft trained an AI on thousands of real people who rated phone call quality from 1 (terrible) to 5 (crystal clear). When the system uses DNSMOS, it's asking this "robot judge" to give a human-like quality opinion.

#### 2. ONNX Runtime
**What it means:** The engine block.

> If DNSMOS is the "driver" making decisions, ONNX Runtime is the "car engine" that allows the AI to actually run inside our application.

#### 3. DeepFilterNet (The Noise Cleaner)
**What it means:** A neural vacuum cleaner for audio.

> If the Math and the Robot Judge decide your audio is awful (e.g., someone talking while a blender runs in the background), DeepFilterNet steps in, finds the blender sound, and erases it while leaving the human voice intact. Output is saved as lossless WAV.

#### 4. Deepgram Nova-2 (Transcription)
**What it means:** The speech-to-text engine.

> Sends the cleaned audio to Deepgram's highest-accuracy model. Tries cleaned audio first (best signal quality), falls back to raw audio if the cleaned version yields low confidence.

#### 5. Pyannote + Gemini (Speaker Identification)
**What it means:** Who said what?

> Pyannote acoustically locates when each speaker is talking. Gemini reads the actual words to assign the correct role — if someone says "मैं HDFC से बोल रही हूं", it knows that's the Agent.

---

## Project Structure

```
bg_noise_check/
├── src/
│   ├── app.py              # FastAPI server + routing
│   ├── noise_analyzer.py   # Core scoring engine (math + AI)
│   ├── audio_cleaner.py    # DeepFilterNet noise suppressor
│   ├── transcriber_pro.py  # Deepgram transcription + hallucination guard
│   └── diarizer.py         # Pyannote acoustic diarization (lazy-loaded)
├── static/
│   ├── index.html          # Web UI
│   ├── css/style.css
│   └── js/main.js
├── data/
│   ├── uploads/            # Temporary raw uploads (gitignored)
│   └── processed/          # DeepFilterNet cleaned WAV outputs (gitignored)
├── uploads/
│   └── samples/            # Test audio samples
├── requirements.txt
└── .env                    # API keys (gitignored)
```

---

## Setup & Running

### 1. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```
DEEPGRAM_API_KEY=your_deepgram_key
GEMINI_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
```

### 3. Start the server
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8766
```

### 4. Open the UI
Navigate to **http://localhost:8766**

---

## API

### Health Check
```bash
curl http://localhost:8766/api/health
# Returns: {"status": "healthy", "gpu": true}
```

### Process Audio File
```bash
curl -X POST http://localhost:8766/api/process \
  -F "file=@your_audio.mp3"
```

**Response:**
```json
{
  "status": "success",
  "is_reliable": false,
  "reliability_score": 42.3,
  "transcribed_source": "cleaned",
  "transcript_confidence": 0.93,
  "transcript": "Agent: नमस्ते मैं HDFC से...\nCustomer: जी हां...",
  "raw_audio_url": "/data/uploads/...",
  "clean_audio_url": "/data/processed/..._clean.wav"
}
```

---

## Readiness Score Guide

| Score | Meaning |
|---|---|
| **90–100%** | ✅ Crystal clear — send to transcription |
| **75–89%** | ✅ Good — acceptable quality |
| **50–74%** | ⚠️ Borderline — may produce errors |
| **0–49%** | ❌ DO NOT TRANSCRIBE — too degraded |

---

## GPU Acceleration

DeepFilterNet3 and Pyannote both run on **CUDA GPU** (if available). The Pyannote diarization model lazy-loads on the first request to ensure fast server startup.
