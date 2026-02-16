import json
import os
try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    print("❌ Error: 'huggingface_hub' not installed. Run: pip install huggingface_hub")
    exit(1)

KEY_FILE = "hugging-face-key.json"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "models/gpt2"

def main():
    print(f"Reading credentials from {KEY_FILE}...")
    with open(KEY_FILE, 'r') as f:
        creds = json.load(f)
    
    token = creds.get("key")
    if not token:
        print("❌ No key found in credential file!")
        return

    print(f"Authenticating with token: {token[:4]}...")
    try:
        login(token=token, add_to_git_credential=False)
        print("✅ Authenticated successfully")
    except Exception as e:
        print(f"⚠️ Authentication warning: {e}")

    print(f"Downloading {MODEL_NAME} to {OUTPUT_DIR}...")
    try:
        # Only download essential files to keep dataset small
        patterns = ["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"]
        
        path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=patterns,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"]
        )
        print(f"✅ Model downloaded to: {path}")
        
        # Verify essential files
        required_files = ["config.json"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
        
        if missing:
            print(f"❌ Verification failed: Missing {missing}")
            # Optional: clean up or create dummy if strictness allows? No, fail loud.
        else:
            print("✅ Verification successful: All critical files present.")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()
