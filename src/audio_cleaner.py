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

    def clean_audio(self, input_path, output_path=None, attenuation_db=50):
        """
        Enhances the audio and saves it to a new file.
        Returns the output path.
        attenuation_db: The noise suppression limit in dB (e.g., 20 for ASR, 60 for listening).
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = f"{base}_cleaned.wav"

        try:
            # 1. Robust Load Data
            import soundfile as sf
            audio_np, orig_sr = sf.read(input_path)
            
            # Convert to mono
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
            
            # INPUT NORMALIZATION: Ensure speech isn't too faint for the model
            rms = np.sqrt(np.mean(audio_np**2))
            if rms < 0.01 and rms > 1e-6:
                # Target an RMS of 0.05 for processing
                audio_np = audio_np * (0.05 / rms)

            # BETTER RESAMPLING: resample_poly is better for upsampling telco audio (8k -> 48k)
            target_sr = self.df_state.sr()
            if orig_sr != target_sr:
                # Use resample_poly which uses a polyphase filter (higher quality than Fourier)
                import scipy.signal
                audio_np = scipy.signal.resample_poly(audio_np, target_sr, orig_sr)
            
            # Convert to torch tensor
            audio = torch.from_numpy(audio_np.astype(np.float32))
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            
            # 2. Pre-Processing: High-Pass Filter (Strip rumble < 80Hz)
            audio = self.apply_high_pass_filter(audio, target_sr, cutoff=80)
            
            # 3. Neural Enhancement: Variable Attenuation
            # Use attenuation_db parameter: 20dB for ASR safety, 60dB for human purity.
            enhanced = enhance(self.model, self.df_state, audio, atten_lim_db=attenuation_db)
            
            # 4. Post-Processing: Normalization
            max_val = torch.max(torch.abs(enhanced))
            if max_val > 1e-6:
                target_peak = 0.90 
                enhanced = enhanced * (target_peak / max_val)
            
            # 5. Save
            save_audio(output_path, enhanced.cpu(), target_sr)
            return output_path
        except Exception as e:
            print(f"DeepFilterNet Cleanup Error: {e}")
            import traceback
            traceback.print_exc()
            return input_path
