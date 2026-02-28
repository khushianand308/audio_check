import torch
from df.enhance import init_df, enhance, load_audio, save_audio
import os

class AudioCleaner:
    """
    Production-grade audio enhancement using DeepFilterNet.
    Suppresses background noise (hiss, chatter, hum) while preserving speech.
    """
    def __init__(self, model_type="none"):
        # init_df will download/load the best available model
        # Using cpu to avoid GPU memory overhead in some environments, 
        # but DeepFilterNet is very fast on CPU.
        self.model, self.df_state, _ = init_df()

    def clean_audio(self, input_path, output_path=None):
        """
        Enhances the audio and saves it to a new file.
        Returns the output path.
        """
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_cleaned{ext}"

        try:
            # Load
            audio, _ = load_audio(input_path, sr=self.df_state.sr())
            
            # Enhance
            enhanced = enhance(self.model, self.df_state, audio)
            
            # Save
            save_audio(output_path, enhanced, self.df_state.sr())
            return output_path
        except Exception as e:
            print(f"DeepFilterNet Cleanup Error: {e}")
            return input_path # Fallback to original if enhancement fails
