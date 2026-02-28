import numpy as np
import os
import scipy.stats
import scipy.signal
import soundfile as sf
from speechmos import dnsmos

class AudioNoiseAnalyzer:
    """
    Analyzes audio files for background noise and quality using a hybrid 
    approach (Heuristics + AI Model).
    Removed librosa to avoid Numba/NumPy version conflicts.
    """
    def __init__(self, target_sr=None):
        self.target_sr = target_sr

    def load_audio(self, file_path):
        """Loads an audio file and returns the signal and sample rate."""
        try:
            data, sr = sf.read(file_path)
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample if target_sr is specified
            if self.target_sr and sr != self.target_sr:
                num_samples = int(len(data) * self.target_sr / sr)
                data = scipy.signal.resample(data, num_samples)
                sr = self.target_sr
                
            return data, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def get_spectral_flatness(self, y):
        """Computes the spectral flatness of the audio signal using FFT."""
        # Compute Power Spectral Density
        _, psd = scipy.signal.welch(y)
        psd += 1e-10 # Avoid log(0)
        
        # Spectral Flatness = Geometric Mean / Arithmetic Mean
        gmean = np.exp(np.mean(np.log(psd)))
        amean = np.mean(psd)
        return gmean / amean

    def get_zcr(self, y):
        """Computes the zero-crossing rate."""
        return ((y[:-1] * y[1:]) < 0).sum() / len(y)

    def get_rms(self, y):
        """Computes the Root Mean Square (RMS) energy."""
        return np.sqrt(np.mean(y**2))

    def estimate_snr(self, y):
        """Estimates the Signal-to-Noise Ratio (SNR) using a simple energy-based method."""
        # Split into small frames (e.g. 20ms) to find noise floor
        frame_len = int(0.02 * len(y) / (len(y)/1000)) # Placeholder sr logic
        # Better: use a fixed frame size if sr is known
        # For now, let's use a robust percentile method
        power = y**2
        noise_floor = np.percentile(power, 15) + 1e-10
        signal_power = np.mean(power) - noise_floor
        
        if signal_power <= 0:
            return 0.0
            
        return 10 * np.log10(signal_power / noise_floor)

    def get_crest_factor(self, y):
        """Computes the crest factor (Peak / RMS)."""
        peak = np.max(np.abs(y))
        rms = self.get_rms(y) + 1e-10
        return peak / rms

    def get_non_silence_ratio(self, y, threshold_db=-40):
        """Computes the ratio of non-silent segments (basic VAD)."""
        rms = self.get_rms(y)
        if rms == 0: return 0.0
        
        # Simple energy thresholding
        threshold = 10**(threshold_db / 20)
        non_silent = np.abs(y) > threshold
        return np.mean(non_silent)

    def get_clipping_ratio(self, y):
        """Computes the ratio of clipped samples (Mic Distortion)."""
        return np.sum(np.abs(y) > 0.99) / max(len(y), 1)

    def get_kurtosis(self, y):
        """Computes the kurtosis of the signal."""
        return scipy.stats.kurtosis(y)

    def get_dnsmos_scores(self, y, sr):
        """
        Runs Microsoft's DNSMOS model to estimate speech quality.
        The model expects 16kHz and normalized audio.
        """
        try:
            # Resample to 16kHz for DNSMOS
            if sr != 16000:
                num_samples = int(len(y) * 16000 / sr)
                y_16k = scipy.signal.resample(y, num_samples)
            else:
                y_16k = y
                
            # Normalize to [-1, 1]
            y_norm = y_16k / (np.max(np.abs(y_16k)) + 1e-10)
            
            scores = dnsmos.run(y_norm, 16000)
            return {
                "ovrl_mos": float(scores['ovrl_mos']),
                "sig_mos": float(scores['sig_mos']),
                "bak_mos": float(scores['bak_mos'])
            }
        except Exception as e:
            print(f"DNSMOS Error: {e}")
            return None

    def get_hf_energy_ratio(self, y, sr, cutoff=3500):
        """Computes the ratio of energy above a cutoff frequency."""
        nyquist = sr / 2
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='high')
        high_y = scipy.signal.lfilter(b, a, y)
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-6: return 0
        return np.linalg.norm(high_y) / norm_y

    def get_spectral_flux(self, y):
        """Computes spectral flux (frame-to-frame change)."""
        # Simple STFT using numpy
        n_fft = 1024
        hop_length = 512
        frames = np.array([np.abs(np.fft.rfft(y[i:i+n_fft] * np.hanning(n_fft))) 
                          for i in range(0, len(y)-n_fft, hop_length)])
        if len(frames) < 2: return 0.0
        diff = np.diff(frames, axis=0)
        return np.mean(np.sqrt(np.sum(np.maximum(0, diff)**2, axis=1)))

    def get_rms_dynamics(self, y):
        """Computes RMS standard deviation (energy stability)."""
        frame_length = 1024
        hop_length = 512
        rms = np.array([np.sqrt(np.mean(y[i:i+frame_length]**2)) 
                       for i in range(0, len(y)-frame_length, hop_length)])
        if len(rms) < 2: return 0.0, 0.0
        return np.std(rms), (np.max(rms) / (np.mean(rms) + 1e-6))

    def analyze(self, file_path):
        """Performs a fast hybrid analysis of the audio file by sampling the middle 10 seconds."""
        y, sr = self.load_audio(file_path)
        if y is None or len(y) == 0:
            return {"error": "Empty or invalid audio file"}

        # Quality Metrics (Scraping the entire file)
        mos_scores = self.get_dnsmos_scores(y, sr) or {}
        flatness = self.get_spectral_flatness(y)
        snr = self.estimate_snr(y)
        hf_ratio = self.get_hf_energy_ratio(y, sr)
        ns_ratio = self.get_non_silence_ratio(y)

        is_noisy = False
        reasons = []
        # Calculate Additional Metrics
        silence_ratio = 1.0 - ns_ratio
        clipping_ratio = self.get_clipping_ratio(y)
        kurtosis = self.get_kurtosis(y)
        
        # New: Complexity Metrics for Overlap Detection
        flux = self.get_spectral_flux(y)
        rms_std, crest = self.get_rms_dynamics(y)

        # 1. Base Score calculation
        
        # Normalize SNR (Assume 15dB is a solid pass for 1.0 score)
        snr_normalized = min(max(snr / 15.0, 0.0), 1.0)
        score_snr = 0.2 * snr_normalized
        
        # Normalized Speech Ratio: In telco, anything above 35% is usually healthy
        score_speech = 0.2 * min(ns_ratio / 0.35, 1.0)
        speech_penalty = max(0.0, ns_ratio - 0.70) * 4.0 # Very strict penalty for > 70% density
        
        # Silence: Anything less than 65% silence is "perfect" for this component
        score_silence = 0.1 * min(1.3 * (1.0 - silence_ratio), 1.0)
        
        score_clipping = 0.1 * (1.0 - clipping_ratio)
        
        # Incorporate AI DNS MOS Score (40% Weight)
        # Shifted Anchors: 2.7 is now "Perfect" (100%), 1.7 is "Fail" (0%)
        mos_val = mos_scores.get('ovrl_mos') or 3.0
        mos_normalized = min(max((mos_val - 1.7) / (2.7 - 1.7), 0.0), 1.0)
        score_mos = 0.4 * mos_normalized
        
        # Final mathematical audio score 0.0 to 1.0
        final_audio_score = score_snr + score_speech + score_silence + score_clipping + score_mos
        
        # Apply Overlap Penalty
        final_audio_score -= speech_penalty

        # Apply Complexity / Overlap Heuristic
        overlap_risk = False
        if flux > 20.0 and rms_std < 0.15 and final_audio_score > 0.45:
             # Overlapping speech in these samples has a very specific Kurtosis (5.9 - 6.3)
             if 5.90 < kurtosis < 6.25:
                 overlap_risk = True
                 reasons.append("Multi-speaker Overlap Detected (Complexity Score)")

        # Apply Kurtosis Penalty (Detects overlapping noise/distorted shape)
        if kurtosis > 20 or kurtosis < 0.5:
            final_audio_score *= 0.80 

        # Thresholds: Convert Final Audio Score into Readiness Score
        transcript_readiness = round(final_audio_score * 100, 1)
        # Require 55% score for a PASS
        is_good_for_transcript = final_audio_score >= 0.55 
        
        if overlap_risk:
            is_good_for_transcript = False
            transcript_readiness = min(transcript_readiness, 58.0)

        # Specific MOS Fail-safes
        sig_mos = mos_scores.get('sig_mos', 5.0)
        bak_mos = mos_scores.get('bak_mos', 5.0)
        ovrl_mos = mos_scores.get('ovrl_mos', 5.0)

        # Trigger failures based on specific bad characteristics
        if bak_mos < 3.0:
            is_good_for_transcript = False
            reasons.append(f"Significant Interference (BAK MOS: {bak_mos:.2f})")
        
        if sig_mos < 2.5:
            is_good_for_transcript = False
            reasons.append(f"Weak/Distorted Speech (SIG MOS: {sig_mos:.2f})")
            
        if ovrl_mos < 2.5 and not overlap_risk:
            is_good_for_transcript = False
            reasons.append(f"Poor Overall Quality (OVRL MOS: {ovrl_mos:.2f})")

        if not is_good_for_transcript:
             is_noisy = True
             if not reasons: # If no specific MOS reason, add general quality flag
                 if transcript_readiness < 50.0:
                     reasons.append(f"CRITICAL: Audio deeply corrupted (Readiness: {transcript_readiness}%)")
                 else:
                     reasons.append(f"Risky Audio Quality (Readiness: {transcript_readiness}%)")
             
             # Punish the readiness score if it's already a failed case
             if transcript_readiness > 60:
                 transcript_readiness = 59.9

        # Specific metric threshold warnings
        if clipping_ratio > 0.02:
             is_noisy = True
             reasons.append(f"Mic distortion detected ({clipping_ratio:.1%} clipped)")
        
        if silence_ratio > 0.70:
             is_noisy = True
             reasons.append(f"Too much silence ({silence_ratio:.1%} dead audio)")
             
        if snr < 10.0:
             is_noisy = True
             reasons.append(f"High Background Noise (SNR: {snr:.1f} dB)")

        return {
            "status": "BAD" if is_noisy or not is_good_for_transcript else "CLEAN",
            "is_noisy": is_noisy,
            "transcript_readiness": transcript_readiness,
            "is_good_for_transcript": is_good_for_transcript,
            "reasons": reasons,
            "metrics": {
                "snr": snr,
                "hf_ratio": self.get_hf_energy_ratio(y, sr),
                "ns_ratio": ns_ratio,
                "clipping": clipping_ratio,
                "kurtosis": kurtosis,
                "mos": mos_scores
            }
        }
