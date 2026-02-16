"""
AIMO Dataset Download & Fine-tuning Guide
==========================================

This module provides utilities to download AIMO datasets from Kaggle
and set up the fine-tuning framework.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleDatasetDownloader:
    """
    Download AIMO datasets from Kaggle competition.
    
    Requirements:
    - Kaggle API installed: pip install kaggle
    - Kaggle credentials: ~/.kaggle/kaggle.json
    
    Setup Instructions:
    1. Go to: https://www.kaggle.com/settings/account
    2. Click "Create New API Token"
    3. Save kaggle.json to ~/.kaggle/
    4. Run: chmod 600 ~/.kaggle/kaggle.json
    """
    
    COMPETITION_NAME = "aimo-progress-prize-2024"
    DATASETS = {
        "aimo1_problems": "AIMO 2023 Problems",
        "aimo2_problems": "AIMO 2024 Problems",
        "aimo3_public_test": "AIMO 3 Public Test Set"
    }
    
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        self.kaggle_credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    
    def check_kaggle_credentials(self) -> bool:
        """Check if Kaggle credentials are configured."""
        if not self.kaggle_credentials_path.exists():
            logger.error(f"❌ Kaggle credentials not found: {self.kaggle_credentials_path}")
            return False
        logger.info(f"✅ Kaggle credentials found: {self.kaggle_credentials_path}")
        return True
    
    def check_kaggle_cli(self) -> bool:
        """Check if Kaggle CLI is installed."""
        try:
            result = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"✅ Kaggle CLI installed: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.error("❌ Kaggle CLI not installed")
        return False
    
    def download_competition_files(self) -> Dict[str, bool]:
        """
        Download all competition files.
        
        Returns:
            Dict mapping dataset names to download success status
        """
        results = {}
        
        print("\n" + "="*80)
        print("📥 DOWNLOADING AIMO DATASETS FROM KAGGLE")
        print("="*80)
        
        if not self.check_kaggle_credentials():
            print("\n❌ Setup Kaggle Credentials:")
            print("   1. Visit: https://www.kaggle.com/settings/account")
            print("   2. Click 'Create New API Token'")
            print("   3. Place kaggle.json in ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return {name: False for name in self.DATASETS.keys()}
        
        if not self.check_kaggle_cli():
            print("\n❌ Install Kaggle CLI:")
            print("   pip install kaggle")
            return {name: False for name in self.DATASETS.keys()}
        
        # Download all files from competition
        print(f"\n📥 Downloading from competition: {self.COMPETITION_NAME}")
        print(f"   Target directory: {self.dataset_dir.absolute()}")
        
        try:
            result = subprocess.run(
                [
                    "kaggle", "competitions", "download",
                    "-c", self.COMPETITION_NAME,
                    "-p", str(self.dataset_dir),
                    "--quiet"
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("✅ Download completed successfully")
                print("\n✅ Files downloaded successfully!")
                
                # Check which files exist
                for filename in ["aimo1_problems.csv", "aimo2_problems.csv", "aimo3_public_test.csv"]:
                    file_path = self.dataset_dir / filename
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        results[filename.replace(".csv", "")] = True
                        print(f"   ✅ {filename} ({size_mb:.2f} MB)")
                    else:
                        results[filename.replace(".csv", "")] = False
                        print(f"   ❌ {filename} (not found)")
            else:
                logger.error(f"Download failed: {result.stderr}")
                print(f"\n❌ Download failed: {result.stderr}")
                return {name: False for name in self.DATASETS.keys()}
        
        except subprocess.TimeoutExpired:
            logger.error("Download timed out (> 5 minutes)")
            print("❌ Download timed out")
            return {name: False for name in self.DATASETS.keys()}
        except Exception as e:
            logger.error(f"Download error: {e}")
            print(f"❌ Download error: {e}")
            return {name: False for name in self.DATASETS.keys()}
        
        return results
    
    def verify_downloads(self) -> Dict[str, bool]:
        """Verify downloaded files."""
        print("\n" + "="*80)
        print("✓ VERIFYING DOWNLOADS")
        print("="*80)
        
        status = {}
        
        for name, description in self.DATASETS.items():
            # Map dataset name to CSV filename
            if "aimo1" in name:
                filename = "aimo1_problems.csv"
            elif "aimo2" in name:
                filename = "aimo2_problems.csv"
            else:
                filename = "aimo3_public_test.csv"
            
            filepath = self.dataset_dir / filename
            
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                status[name] = True
                print(f"✅ {name:25} - {description:30} ({size_mb:.2f} MB)")
            else:
                status[name] = False
                print(f"❌ {name:25} - {description:30} (MISSING)")
        
        return status
    
    def get_setup_instructions(self):
        """Print setup instructions."""
        print("\n" + "="*80)
        print("📋 SETUP INSTRUCTIONS")
        print("="*80)
        
        print("\n1️⃣ INSTALL KAGGLE CLI")
        print("   pip install kaggle")
        
        print("\n2️⃣ GET API CREDENTIALS")
        print("   a) Go to: https://www.kaggle.com/settings/account")
        print("   b) Scroll to 'API' section")
        print("   c) Click 'Create New API Token'")
        print("   d) This downloads kaggle.json")
        
        print("\n3️⃣ PLACE CREDENTIALS")
        print(f"   mkdir -p ~/.kaggle")
        print(f"   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print(f"   chmod 600 ~/.kaggle/kaggle.json")
        
        print("\n4️⃣ DOWNLOAD DATASETS")
        print("   python src/kaggle_downloader.py")
        
        print("\n" + "="*80)


def download_aimo_datasets():
    """Download AIMO datasets."""
    
    downloader = KaggleDatasetDownloader(dataset_dir="datasets")
    
    # Show setup instructions
    downloader.get_setup_instructions()
    
    # Try to download
    print("\n" + "="*80)
    print("🚀 ATTEMPTING DOWNLOAD")
    print("="*80)
    
    results = downloader.download_competition_files()
    
    # Verify downloads
    status = downloader.verify_downloads()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    all_success = all(status.values())
    
    if all_success:
        print("\n✅ ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        print("\nNext steps:")
        print("   1. Run: python src/task_3_1_datasets.py")
        print("      This will validate and prepare the datasets")
        print("   2. Proceed with Task 3.2: Create training data for fine-tuning")
    else:
        print("\n⚠️ SOME DATASETS MISSING")
        print("\nTroubleshooting:")
        print("   • Verify Kaggle credentials are correct")
        print("   • Check internet connection")
        print("   • Ensure you're joined to the competition")
        print("   • Try manual download from:")
        print("     https://www.kaggle.com/c/aimo-progress-prize-2024/data")


if __name__ == "__main__":
    download_aimo_datasets()
