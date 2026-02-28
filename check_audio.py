import sys
import os
import json
import argparse
from noise_analyzer import AudioNoiseAnalyzer
from transcript_analyzer import TranscriptAnalyzer
from transcriber import WhisperTranscriber

def main():
    parser = argparse.ArgumentParser(description="Analyze audio and generate validated transcripts (Two-Layer Validation).")
    parser.add_argument("path", help="Path to an audio file or directory.")
    parser.add_argument("--transcript", help="Optional: Transcript text to validate against the audio (skips auto-ASR).")
    parser.add_argument("--transcript_file", help="Optional: Path to a text file containing the transcript.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    parser.add_argument("--model", default="tiny", help="Whisper model size (tiny, base, small, medium, large). Default: tiny.")
    
    args = parser.parse_args()
    
    audio_analyzer = AudioNoiseAnalyzer()
    text_analyzer = TranscriptAnalyzer()
    
    # Lazily initialize transcriber only if needed
    transcriber = None

    results = []
    
    def process_file(file_path, manual_transcript=None):
        nonlocal transcriber
        audio_res = audio_analyzer.analyze(file_path)
        if "error" in audio_res:
            return audio_res
            
        final_res = {
            "audio": audio_res,
            "transcript_text": None,
            "transcript_analysis": None,
            "is_reliable": not audio_res["is_noisy"],
            "reasons": audio_res["reasons"][:]
        }
        
        # Determine transcript to use
        transcript_to_validate = manual_transcript
        
        # If no manual transcript provided AND audio is not critically bad, try auto-transcription
        # We allow transcription even if audio has minor issues to see if NLP layer catches bigger problems
        if not transcript_to_validate:
            # Check for critical audio failure (e.g. mostly silent or extreme noise)
            if audio_res.get('non_silence_ratio', 1.0) < 0.05:
                final_res["is_reliable"] = False
                final_res["reasons"].append("Audio too silent for transcription")
            else:
                if transcriber is None:
                    transcriber = WhisperTranscriber(model_size=args.model)
                
                print(f"Transcribing {os.path.basename(file_path)}...")
                trans_res = transcriber.transcribe(file_path)
                if trans_res:
                    transcript_to_validate = trans_res["text"]
                    final_res["transcript_text"] = transcript_to_validate
                else:
                    final_res["is_reliable"] = False
                    final_res["reasons"].append("Transcription failed")

        if transcript_to_validate:
            text_res = text_analyzer.analyze(transcript_to_validate)
            final_res["transcript_analysis"] = text_res
            final_res["transcript_text"] = transcript_to_validate
            
            # If transcript is bad (hallucinations/loops), mark as unreliable
            if not text_res["is_reliable"]:
                final_res["is_reliable"] = False
                final_res["reasons"].extend([f"Transcript: {r}" for r in text_res["reasons"]])
        
        return final_res

    # Load manual transcript if provided
    input_transcript = args.transcript
    if args.transcript_file and os.path.exists(args.transcript_file):
        with open(args.transcript_file, 'r') as f:
            input_transcript = f.read()

    if os.path.isfile(args.path):
        try:
            results.append(process_file(args.path, input_transcript))
        except Exception as e:
            results.append({"filename": os.path.basename(args.path), "error": str(e)})
    elif os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    full_path = os.path.join(root, file)
                    try:
                        results.append(process_file(full_path, None))
                    except Exception as e:
                        results.append({"filename": file, "error": str(e)})
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'Filename':<30} | {'Status':<10} | {'OVRL':<6} | {'BAK':<6}")
        print("-" * 65)
        for res in results:
            if "error" in res:
                print(f"{res.get('filename', 'Unknown'):<30} | ERROR: {res['error']}")
                continue
            
            audio = res["audio"]
            status_str = "CLEAN" if res['is_reliable'] else "BAD"
            ovrl = audio.get('ovrl_mos') or 0.0
            bak = audio.get('bak_mos') or 0.0
            
            print(f"{audio['filename'][:30]:<30} | {status_str:<10} | {ovrl:<6.2f} | {bak:<6.2f}")
            if res['is_reliable'] and res['transcript_text']:
                print(f"  Transcript: \"{res['transcript_text'][:100]}...\"")
            
            if not res['is_reliable'] and res['reasons']:
                for reason in res['reasons']:
                    print(f"  > {reason}")
            print("-" * 65)

if __name__ == "__main__":
    main()
