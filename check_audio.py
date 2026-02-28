import sys
import os
import json
import argparse
from noise_analyzer import AudioNoiseAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze audio noise and MOS scores (Step 1 Audit).")
    parser.add_argument("path", help="Path to an audio file or directory.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    
    args = parser.parse_args()
    
    audio_analyzer = AudioNoiseAnalyzer(target_sr=16000)
    
    results = []
    
    def process_file(file_path):
        audio_res = audio_analyzer.analyze(file_path)
        if "error" in audio_res:
            return {"filename": os.path.basename(file_path), "error": audio_res["error"]}
            
        return {
            "filename": os.path.basename(file_path),
            "audio": audio_res,
            "is_reliable": not audio_res.get("is_noisy", False),
            "reasons": audio_res.get("reasons", [])
        }

    if os.path.isfile(args.path):
        try:
            results.append(process_file(args.path))
        except Exception as e:
            results.append({"filename": os.path.basename(args.path), "error": str(e)})
    elif os.path.isdir(args.path):
        for root, _, files in os.walk(args.path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                    full_path = os.path.join(root, file)
                    try:
                        results.append(process_file(full_path))
                    except Exception as e:
                        results.append({"filename": file, "error": str(e)})
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\n{'Filename':<35} | {'Status':<10} | {'OVRL':<6} | {'SIG':<6} | {'BAK':<6}")
        print("-" * 80)
        for res in results:
            if "error" in res:
                print(f"{res.get('filename', 'Unknown'):<35} | ERROR: {res['error']}")
                continue
            
            audio = res["audio"]
            status_str = "CLEAN" if res['is_reliable'] else "BAD"
            ovrl = audio.get('ovrl_mos') or 0.0
            sig = audio.get('sig_mos') or 0.0
            bak = audio.get('bak_mos') or 0.0
            
            print(f"{res['filename'][:35]:<35} | {status_str:<10} | {ovrl:<6.2f} | {sig:<6.2f} | {bak:<6.2f}")
            
            if not res['is_reliable'] and res['reasons']:
                for reason in res['reasons']:
                    print(f"  > {reason}")
            print("-" * 80)

if __name__ == "__main__":
    main()
