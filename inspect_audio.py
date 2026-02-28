import librosa
import numpy as np
import scipy.stats

def inspect_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    rms = librosa.feature.rms(y=y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Statistical measures
    kurtosis = scipy.stats.kurtosis(y)
    skewness = scipy.stats.skew(y)
    
    print(f"File: {file_path}")
    print(f"Sample Rate: {sr}")
    print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
    print(f"Mean RMS: {np.mean(rms):.6f}")
    print(f"Max RMS: {np.max(rms):.6f}")
    print(f"Min RMS: {np.min(rms):.6f}")
    print(f"Mean Flatness: {np.mean(flatness):.6f}")
    print(f"Mean ZCR: {np.mean(zcr):.6f}")
    print(f"Mean Centroid: {np.mean(centroid):.2f}")
    print(f"Mean Rolloff: {np.mean(rolloff):.2f}")
    print(f"Kurtosis: {kurtosis:.2f}")
    print(f"Skewness: {skewness:.2f}")
    
    # Peak to RMS Ratio (Crest Factor)
    peak = np.max(np.abs(y))
    crest_factor = peak / (np.sqrt(np.mean(y**2)) + 1e-10)
    print(f"Crest Factor: {crest_factor:.2f}")

if __name__ == "__main__":
    import sys
    inspect_audio(sys.argv[1])
