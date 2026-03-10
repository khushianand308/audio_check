import torch
from df.enhance import init_df, enhance, load_audio, save_audio
import os
import scipy.signal
import numpy as np

class AudioCleaner:
    """
    Surgical-grade audio enhancement using DeepFilterNet + Butterworth Filters.
    Suppresses background noise aggressively while stripping low-end rumble.
    """
    def __init__(self, model_type="none"):
        # Initialize DeepFilterNet (uses CUDA if available)
        self.model, self.df_state, _ = init_df()

    def apply_high_pass_filter(self, audio, sr, cutoff=100):
        """Applies a Butterworth high-pass filter to remove DC offset and low hum."""
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        # 5th order for sharp cutoff
        b, a = scipy.signal.butter(5, normal_cutoff, btype='high', analog=False)
        
        # Convert torch tensor to numpy for scipy
        audio_np = audio.cpu().numpy()
        filtered_np = scipy.signal.filtfilt(b, a, audio_np)
        return torch.from_numpy(filtered_np.copy().astype(np.float32)).to(audio.device)

    def clean_audio(self, input_path, output_path=None):
        """
        Enhances the audio and saves it to a new file.
        Returns the output path.
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_cleaned.wav"

        try:
            # 1. Load Data
            audio, _ = load_audio(input_path, sr=self.df_state.sr())
            
            # 2. Pre-Processing: High-Pass Filter (Strip rumble < 100Hz)
            audio = self.apply_high_pass_filter(audio, self.df_state.sr(), cutoff=100)
            
            # 3. Neural Enhancement: Aggressive 100dB Attenuation
            # Bumped from 10dB to 100dB to completely zero out stubborn noise.
            enhanced = enhance(self.model, self.df_state, audio, atten_lim_db=100)
            
            # 4. Post-Processing: Surgical Normalization
            # Target peak = -1dB (0.89) for maximum punch without clipping
            max_val = torch.max(torch.abs(enhanced))
            if max_val > 1e-6:
                target_peak = 0.89 
                enhanced = enhanced * (target_peak / max_val)
            
            # 5. Save (Lossless WAV)
            save_audio(output_path, enhanced, self.df_state.sr())
            return output_path
        except Exception as e:
            print(f"DeepFilterNet Cleanup Error: {e}")
            return input_path
