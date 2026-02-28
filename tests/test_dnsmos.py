import librosa
import numpy as np
from speechmos import dnsmos
import os

def test_dnsmos(file_path):
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    y, sr = librosa.load(file_path, sr=16000) # DNSMOS expects 16kHz
    
    # Normalize to [-1, 1]
    y = y / (np.max(np.abs(y)) + 1e-10)
    
    # Run DNSMOS
    scores = dnsmos.run(y, sr)
    
    print(f"Overall MOS (1-5): {scores['ovrl_mos']:.2f}")
    print(f"Signal MOS (1-5):  {scores['sig_mos']:.2f}")
    print(f"Background MOS (1-5): {scores['bak_mos']:.2f}")
    
    return scores

if __name__ == "__main__":
    samples = [
        "/home/ubuntu/bg_noise_check/audio_sample_2.mp3",
        "/home/ubuntu/bg_noise_check/audio_sample_3.mp3"
    ]
    
    for s in samples:
        if os.path.exists(s):
            test_dnsmos(s)
        else:
            print(f"File not found: {s}")
