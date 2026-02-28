from transformers import pipeline
import librosa
import numpy as np
import os
import torch

def test_hf_noise_detection(file_path):
    print(f"\nAnalyzing with HF Model: {os.path.basename(file_path)}")
    
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Initialize pipeline
    # The model expects audio classification
    classifier = pipeline("audio-classification", model="Etherll/NoisySpeechDetection-v0.2", device="cpu")
    
    # Split audio into 5-second chunks as classification models often have limits
    chunk_size = 5 * 16000
    results = []
    
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) < 16000: # skip very short end chunks
            continue
        
        # Pipeline expects numpy array
        pred = classifier(chunk)
        results.append(pred)
    
    # Aggregate results
    # pred is a list of dicts like [{'label': 'LABEL_0', 'score': 0.9}, ...]
    # LABEL_0 = Clean, LABEL_1 = Noisy (according to research)
    
    noisy_count = 0
    clean_count = 0
    
    for res in results:
        top_label = res[0]['label']
        if top_label == 'LABEL_1':
            noisy_count += 1
        else:
            clean_count += 1
            
    print(f"Chunks: Noisy={noisy_count}, Clean={clean_count}")
    is_noisy = (noisy_count > 0) # Flag if any chunk is noisy
    
    return is_noisy

if __name__ == "__main__":
    samples = [
        "/home/ubuntu/bg_noise_check/audio_sample_2.mp3",
        "/home/ubuntu/bg_noise_check/audio_sample_3.mp3"
    ]
    
    for s in samples:
        if os.path.exists(s):
            test_hf_noise_detection(s)
        else:
            print(f"File not found: {s}")
