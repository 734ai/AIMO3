#!/usr/bin/env python3
"""
Master Verification Script for AIMO3 Kaggle Project
Run this to verify the entire project state before deployment.
"""
import os
import sys
import subprocess
import json

def run_step(name, command):
    print(f"\n🔹 Running: {name}...")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        else:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"✅ {name} Passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} Failed!")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {name} Error: {e}")
        return False

def check_file(path):
    if os.path.exists(path):
        print(f"✅ Found: {path}")
        return True
    else:
        print(f"❌ Missing: {path}")
        return False

def verify_project():
    print("="*60)
    print("🚀 AIMO3 PROJECT VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # 1. Check Critical Files
    print("\n1️⃣ Checking Critical Files")
    files_to_check = [
        "notebooks/aimo3_kaggle_ready.ipynb",
        "notebooks/kernel-metadata.json",
        "src/config.py",
        "src/pipeline.py",
        "update_notebook.py",
        "test_notebook_logic.py"
    ]
    for f in files_to_check:
        if not check_file(f):
            all_passed = False

    # 2. Verify Notebook Logic
    print("\n2️⃣ Verifying Notebook Logic")
    if not run_step("Notebook Logic Test", [sys.executable, "test_notebook_logic.py"]):
        all_passed = False
        
    # 3. Verify Model Config (Lightweight check)
    print("\n3️⃣ Checking Model Config")
    if os.path.exists("models/gpt2/config.json"):
        print("✅ GPT-2 Model config found")
    else:
        print("⚠️ GPT-2 Model config missing (Warning only - might be downloading)")
        
    print("\n" + "="*60)
    if all_passed:
        print("✅✅ PROJECT READY FOR DEPLOYMENT ✅✅")
        print("Run: kaggle kernels push -p notebooks")
    else:
        print("❌❌ VERIFICATION FAILED - FIX ERRORS BEFORE DEPLOYING ❌❌")
    print("="*60)

if __name__ == "__main__":
    verify_project()
