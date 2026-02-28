import numpy as np
import soundfile as sf
import os
from noise_analyzer import AudioNoiseAnalyzer

def generate_synthetic_audio(filename, duration=3.0, sr=22050, noise_level=0.0):
    """
    Generates a synthetic audio file (a simple sine wave + noise).
    noise_level: 0.0 to 1.0
    """
    t = np.linspace(0, duration, int(sr * duration))
    # A 440Hz sine wave (speech-like tone) - intermittent
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # zero out some parts of the signal to simulate gaps in speech
    mask = np.ones_like(t)
    mask[int(len(t)*0.3):int(len(t)*0.7)] = 0 # silence in the middle
    signal = signal * mask
    
    # Add white noise
    noise = np.random.normal(0, noise_level, len(t))
    audio = signal + noise
    
    # Normalize to avoid clipping
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
        
    sf.write(filename, audio, sr)
    print(f"Generated {filename} (noise level: {noise_level})")

def test_on_synthetic():
    analyzer = AudioNoiseAnalyzer()
    
    # Create test directory
    test_dir = "/tmp/audio_test"
    os.makedirs(test_dir, exist_ok=True)
    
    clean_file = os.path.join(test_dir, "clean.wav")
    noisy_file = os.path.join(test_dir, "noisy.wav")
    
    generate_synthetic_audio(clean_file, noise_level=0.001) # Very low noise
    generate_synthetic_audio(noisy_file, noise_level=0.2)   # High noise
    
    print("\n--- Testing Clean File ---")
    clean_results = analyzer.analyze(clean_file)
    print(f"SNR: {clean_results['snr_db']:.2f} dB")
    print(f"Flatness: {clean_results['spectral_flatness']:.4f}")
    print(f"Is Noisy? {clean_results['is_noisy']}")
    
    print("\n--- Testing Noisy File ---")
    noisy_results = analyzer.analyze(noisy_file)
    print(f"SNR: {noisy_results['snr_db']:.2f} dB")
    print(f"Flatness: {noisy_results['spectral_flatness']:.4f}")
    print(f"Is Noisy? {noisy_results['is_noisy']}")

    # Verification
    assert clean_results['is_noisy'] is False, "Clean file should not be marked as noisy"
    assert noisy_results['is_noisy'] is True, "Noisy file should be marked as noisy"
    assert clean_results['snr_db'] > noisy_results['snr_db'], "Clean file should have higher SNR"
    
    print("\nVERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    test_on_synthetic()
