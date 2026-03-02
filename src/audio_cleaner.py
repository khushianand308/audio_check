import torch
from df.enhance import init_df, enhance, load_audio, save_audio
import os

class AudioCleaner:
    """
    Production-grade audio enhancement using DeepFilterNet.
    Suppresses background noise (hiss, chatter, hum) while preserving speech.
    """
    def __init__(self, model_type="none"):
        # Allow DeepFilterNet to use GPU (CUDA) for faster noise suppression
        self.model, self.df_state, _ = init_df()


    def clean_audio(self, input_path, output_path=None):
        """
        Enhances the audio and saves it to a new file.
        Returns the output path.
        """
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            # Default to .wav for lossless quality after processing
            output_path = f"{base}_cleaned.wav"

        try:
            # Load
            audio, _ = load_audio(input_path, sr=self.df_state.sr())
            
            # Enhance with a 10dB attenuation limit to prevent "robotic" artifacts
            # while still significantly reducing noise.
            enhanced = enhance(self.model, self.df_state, audio, atten_lim_db=10)
            
            # Post-Cleanup Vocal Normalization
            # Ensure the vocal energy is consistently at a clear volume (-3dB peak)
            max_val = torch.max(torch.abs(enhanced))
            if max_val > 0:
                # Target peak = 0.707 (-3dB)
                target_peak = 0.707
                enhanced = enhanced * (target_peak / max_val)
            
            # Save as WAV (PCM_16) for maximum fidelity
            save_audio(output_path, enhanced, self.df_state.sr())
            return output_path
        except Exception as e:
            print(f"DeepFilterNet Cleanup Error: {e}")
            return input_path # Fallback to original if enhancement fails
