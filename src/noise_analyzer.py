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
        # Compute Periodogram
        freqs, psd = scipy.signal.periodogram(y, sr)
        
        hf_mask = freqs > cutoff
        if not np.any(hf_mask):
            return 0.0
            
        hf_energy = np.sum(psd[hf_mask])
        total_energy = np.sum(psd) + 1e-10
        
        return hf_energy / total_energy

    def analyze(self, file_path):
        """Performs a fast hybrid analysis of the audio file by sampling the middle 10 seconds."""
        y, sr = self.load_audio(file_path)
        if y is None or len(y) == 0:
            return {"error": "Empty or invalid audio file"}

        # Quality Metrics (Scraping the entire file)
        mos_scores = self.get_dnsmos_scores(y, sr) or {}
        flatness = self.get_spectral_flatness(y)
        snr = self.estimate_snr(y)
        kurtosis = self.get_kurtosis(y)
        hf_ratio = self.get_hf_energy_ratio(y, sr)
        ns_ratio = self.get_non_silence_ratio(y)

        is_noisy = False
        reasons = []
        # Calculate Additional Metrics
        silence_ratio = 1.0 - ns_ratio
        clipping_ratio = self.get_clipping_ratio(y)

        # 1. Base Score calculation according to recommended formula
        # Final Score = 0.4 * SNR_score + 0.3 * Speech_ratio + 0.2 * (1 - Silence_ratio) + 0.1 * (1 - Clipping_ratio)
        
        # Normalize SNR (Assume 20dB is max score of 1.0, 0dB is 0.0)
        snr_normalized = min(max(snr / 20.0, 0.0), 1.0)
        
        # Calculate raw component scores
        score_snr = 0.3 * snr_normalized
        
        # Penalize Speech Ratio if it's too high (indicates dense overlapping chatter/noise)
        # Optimal telephonic speech ratio is ~30-40%. >50% means no one is breathing or background noise is triggering VAD.
        optimal_speech_ratio = min(ns_ratio, 0.45)
        speech_penalty = max(0.0, ns_ratio - 0.45) * 1.5
        score_speech = 0.2 * optimal_speech_ratio
        
        score_silence = 0.1 * (1.0 - silence_ratio)
        score_clipping = 0.1 * (1.0 - clipping_ratio)
        
        # Incorporate AI DNS MOS Score (0.0 to 1.0 weight) if available
        # DNSMOS rates audio 1 to 5. We normalize that to 0 to 1.
        mos_val = mos_scores.get('ovrl_mos') or 3.0 # Default to moderate if offline
        score_mos = 0.3 * ((mos_val - 1.0) / 4.0)
        
        # Final mathematical audio score 0.0 to 1.0
        final_audio_score = score_snr + score_speech + score_silence + score_clipping + score_mos
        
        # Apply Overlap Penalty
        final_audio_score -= speech_penalty

        # Apply Kurtosis Penalty (Detects overlapping noise/distorted shape)
        # Normal speech has a kurtosis around 3 to 10. Too low/high means noise collision
        if kurtosis > 15 or kurtosis < 1:
            final_audio_score *= 0.85 # 15% penalty for abnormal spectral shape

        # Thresholds: < 0.6 -> Noisy, > 0.80 -> Clean (Made Stricter)
        # Convert Final Audio Score into user-requested Transcript Readiness Score
        transcript_readiness = round(final_audio_score * 100, 1)
        is_good_for_transcript = final_audio_score >= 0.70 # Require 70% perfection due to DNSMOS
        
        # Hard fail if AI specifically detects corrupt signal or loud overlapping background noise
        sig_mos = mos_scores.get('sig_mos', 5.0)
        bak_mos = mos_scores.get('bak_mos', 5.0)
        if sig_mos < 2.5 or bak_mos < 2.0:
            is_good_for_transcript = False
            # Punish the readiness score heavily to reflect the AI MOS fail
            transcript_readiness = min(transcript_readiness, 45.0) 
        
        if not is_good_for_transcript:
             is_noisy = True
             if transcript_readiness < 50.0:
                 reasons.append(f"CRITICAL: Audio deeply corrupted (Readiness: {transcript_readiness}%)")
             else:
                 reasons.append(f"Risky Audio Quality (Readiness: {transcript_readiness}%)")

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
            "filename": os.path.basename(file_path),
            "sample_rate": int(sr),
            "ovrl_mos": mos_scores.get('ovrl_mos'),
            "bak_mos": mos_scores.get('bak_mos'),
            "is_noisy": is_noisy,
            "reasons": reasons,
            "snr_db": float(snr),
            "hf_energy_ratio": float(hf_ratio),
            "non_silence_ratio": float(ns_ratio),
            "silence_ratio": float(silence_ratio),
            "clipping_ratio": float(clipping_ratio),
            "final_audio_score": float(final_audio_score),
            "transcript_readiness": float(transcript_readiness),
            "is_good_for_transcript": bool(is_good_for_transcript)
        }
