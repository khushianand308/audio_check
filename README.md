# AudioSense Pro 🎙️
**Neural Call Audit System** — AI-powered audio quality analysis for call center recordings.

Automatically scores, flags, and cleans phone call recordings before sending them to transcription engines like Deepgram — reducing hallucinations and wasted API spend.

---

## What It Does

Upload any `.mp3` or `.wav` call recording. The system will:

1. **Score the audio quality** (0–100% Transcript Readiness Score)
2. **Flag specific problems** — background noise, overlapping speakers, mic clipping, dead air
3. **Automatically clean noisy audio** using DeepFilterNet (AI noise suppressor)
4. **Compare** the original and cleaned audio side-by-side in the browser
5. **Log a session history** of every file audited

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Server** | FastAPI + Uvicorn |
| **Audio Math** | NumPy, SciPy (SNR, Spectral Flux, Kurtosis, VAD) |
| **AI Quality Score** | Microsoft DNSMOS via `speechmos` + ONNX Runtime |
| **AI Noise Cleaner** | DeepFilterNet3 (GPU-accelerated) |
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

## Project Structure

```
bg_noise_check/
├── src/
│   ├── app.py              # FastAPI server + routing
│   ├── noise_analyzer.py   # Core scoring engine (math + AI)
│   ├── audio_cleaner.py    # DeepFilterNet noise suppressor
│   └── transcriber_pro.py  # Deepgram transcription hook
├── static/
│   ├── index.html          # Web UI
│   ├── css/style.css
│   └── js/main.js
├── data/
│   ├── uploads/            # Temporary raw uploads (gitignored)
│   └── processed/          # DeepFilterNet cleaned outputs (gitignored)
├── uploads/                # Test audio samples
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
Create a `.env` file:
```
DEEPGRAM_API_KEY=your_key_here
```

### 3. Start the server
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8081
```

### 4. Open the UI
Navigate to **http://localhost:8081**

---

## API

### Health Check
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/
# Returns: 200
```

### Process Audio File
```bash
curl -X POST http://localhost:8081/api/process \
  -F "file=@your_audio.mp3"
```

**Response:**
```json
{
  "status": "success",
  "is_reliable": true,
  "reliability_score": 94.5,
  "audio_audit": {
    "status": "CLEAN",
    "metrics": { "snr": 32.1, "mos": { "ovrl_mos": 3.2 } }
  },
  "raw_file_url": "/data/uploads/...",
  "cleaned_file_url": null
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

DeepFilterNet3 runs on **CUDA GPU** (if available) for fast noise suppression. On CPU, expect ~15–20 seconds per minute of audio.
