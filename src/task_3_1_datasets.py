"""
TASK 3.1: Download & Prepare Real Datasets
============================================

This module handles downloading and preparing AIMO datasets for training.

Status: READY TO IMPLEMENT
Location: src/data_preparation.py (extend existing module)
Timeline: 1-2 days
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIOMDatasetManager:
    """
    Manages downloading, loading, and preparing AIMO datasets.
    
    AIMO datasets:
    - AIMO1: ~100 problems from 2023 competition
    - AIMO2: ~100 problems from 2024 competition
    - AIMO3: Public test set (~1000 problems) for evaluation
    """
    
    def __init__(self, data_dir: str = "datasets"):
        """Initialize dataset manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.aimo1_path = self.data_dir / "aimo1_problems.csv"
        self.aimo2_path = self.data_dir / "aimo2_problems.csv"
        self.aimo3_path = self.data_dir / "aimo3_public_test.csv"
        self.combined_path = self.data_dir / "aimo_combined.csv"
    
    def check_datasets_exist(self) -> Dict[str, bool]:
        """Check which datasets are already downloaded."""
        status = {
            "aimo1": self.aimo1_path.exists(),
            "aimo2": self.aimo2_path.exists(),
            "aimo3": self.aimo3_path.exists(),
        }
        return status
    
    def log_dataset_status(self):
        """Log current dataset status."""
        print("\n" + "="*80)
        print("📊 AIMO DATASET STATUS")
        print("="*80)
        
        status = self.check_datasets_exist()
        
        for dataset, exists in status.items():
            symbol = "✅" if exists else "❌"
            print(f"{symbol} {dataset.upper()}: {'READY' if exists else 'MISSING'}")
            
            if dataset == "aimo1" and exists:
                df = pd.read_csv(self.aimo1_path)
                print(f"   └─ {len(df)} problems | Columns: {list(df.columns)}")
            elif dataset == "aimo2" and exists:
                df = pd.read_csv(self.aimo2_path)
                print(f"   └─ {len(df)} problems | Columns: {list(df.columns)}")
            elif dataset == "aimo3" and exists:
                df = pd.read_csv(self.aimo3_path)
                print(f"   └─ {len(df)} problems | Columns: {list(df.columns)}")
        
        print("\n📥 DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print("If datasets are missing, download from Kaggle:")
        print("  1. Visit: https://www.kaggle.com/c/aimo-progress-prize-2024")
        print("  2. Go to Data tab")
        print("  3. Download:")
        print("     • aimo1_problems.csv → datasets/")
        print("     • aimo2_problems.csv → datasets/")
        print("     • aimo3_public_test.csv → datasets/")
        print("  4. Or use Kaggle API:")
        print("     kaggle competitions download aimo-progress-prize-2024")
        print("="*80 + "\n")
    
    def load_aimo1(self) -> Optional[pd.DataFrame]:
        """Load AIMO1 dataset."""
        if not self.aimo1_path.exists():
            logger.warning(f"AIMO1 dataset not found: {self.aimo1_path}")
            return None
        
        df = pd.read_csv(self.aimo1_path)
        logger.info(f"Loaded AIMO1: {len(df)} problems")
        return df
    
    def load_aimo2(self) -> Optional[pd.DataFrame]:
        """Load AIMO2 dataset."""
        if not self.aimo2_path.exists():
            logger.warning(f"AIMO2 dataset not found: {self.aimo2_path}")
            return None
        
        df = pd.read_csv(self.aimo2_path)
        logger.info(f"Loaded AIMO2: {len(df)} problems")
        return df
    
    def load_aimo3(self) -> Optional[pd.DataFrame]:
        """Load AIMO3 public test set."""
        if not self.aimo3_path.exists():
            logger.warning(f"AIMO3 dataset not found: {self.aimo3_path}")
            return None
        
        df = pd.read_csv(self.aimo3_path)
        logger.info(f"Loaded AIMO3 public test: {len(df)} problems")
        return df
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Validate dataset structure and content."""
        print(f"\n🔍 Validating {dataset_name}...")
        
        # Check columns
        required_columns = {"problem_id", "problem", "answer"} or {"problem_id", "problem"}
        
        if dataset_name == "aimo3":
            # AIMO3 test set might not have answers
            required_cols = {"problem_id", "problem"}
        else:
            required_cols = {"problem_id", "problem", "answer"}
        
        has_required = all(col in df.columns for col in required_cols)
        
        if not has_required:
            logger.error(f"Missing required columns. Found: {list(df.columns)}")
            return False
        
        # Check for nulls
        null_cols = df.columns[df.isnull().any()]
        if len(null_cols) > 0:
            logger.warning(f"Null values found in: {list(null_cols)}")
        
        # Check data types
        print(f"  ✅ Columns: {list(df.columns)}")
        print(f"  ✅ Rows: {len(df)}")
        print(f"  ✅ Data types: {dict(df.dtypes)}")
        
        # Sample
        print(f"  📌 Sample problem:")
        sample = df.iloc[0]
        print(f"     ID: {sample.get('problem_id', 'N/A')}")
        print(f"     Problem: {str(sample.get('problem', 'N/A'))[:80]}...")
        if "answer" in df.columns:
            print(f"     Answer: {sample.get('answer', 'N/A')}")
        
        return True
    
    def combine_datasets(self) -> Optional[pd.DataFrame]:
        """Combine AIMO1 and AIMO2 for training."""
        print("\n🔀 Combining AIMO1 and AIMO2 datasets...")
        
        aimo1 = self.load_aimo1()
        aimo2 = self.load_aimo2()
        
        if aimo1 is None or aimo2 is None:
            logger.error("Cannot combine: AIMO1 or AIMO2 not available")
            return None
        
        # Ensure consistent column names (handle missing problem_id)
        aimo1_cols = ["problem_id", "problem", "answer"] if "problem_id" in aimo1.columns else ["problem", "answer"]
        aimo2_cols = ["problem_id", "problem", "answer"] if "problem_id" in aimo2.columns else ["problem", "answer"]
        
        aimo1 = aimo1[aimo1_cols]
        aimo2 = aimo2[aimo2_cols]
        
        combined = pd.concat([aimo1, aimo2], ignore_index=True)
        logger.info(f"Combined: {len(combined)} total problems")
        
        # Save combined dataset
        combined.to_csv(self.combined_path, index=False)
        logger.info(f"Saved combined dataset: {self.combined_path}")
        
        return combined
    
    def get_dataset_statistics(self, df: pd.DataFrame, name: str = "Dataset"):
        """Generate statistics about the dataset."""
        print(f"\n📈 {name} Statistics:")
        print(f"  Total problems: {len(df)}")
        
        if "answer" in df.columns:
            answers = pd.to_numeric(df["answer"], errors="coerce")
            print(f"  Answer range: {answers.min():.0f} - {answers.max():.0f}")
            print(f"  Answer mean: {answers.mean():.1f}")
            print(f"  Answer median: {answers.median():.1f}")
        
        print(f"  Problem length stats (chars):")
        problem_lens = df["problem"].str.len()
        print(f"    Min: {problem_lens.min()}")
        print(f"    Max: {problem_lens.max()}")
        print(f"    Mean: {problem_lens.mean():.1f}")
    
    def prepare_for_training(self, 
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.2,
                           seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\n📚 Preparing data for training...")
        
        # Load and combine AIMO1 + AIMO2
        combined = self.combine_datasets()
        if combined is None:
            logger.error("Cannot prepare training data: combined dataset unavailable")
            return None, None, None
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # Split
        split_point = int(len(combined) * train_ratio)
        train_df = combined[:split_point]
        val_df = combined[split_point:]
        
        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
        
        # Load test set (AIMO3)
        test_df = self.load_aimo3()
        
        return train_df, val_df, test_df


def demonstrate_task_3_1():
    """Demonstrate Task 3.1: Dataset preparation."""
    
    print("\n" + "="*80)
    print("🚀 TASK 3.1: Download & Prepare Real Datasets")
    print("="*80)
    
    # Initialize manager
    manager = AIOMDatasetManager(data_dir="datasets")
    
    # Check status
    manager.log_dataset_status()
    
    # Check which datasets are available
    status = manager.check_datasets_exist()
    
    if status["aimo1"] and status["aimo2"]:
        # Load and validate
        print("\n✅ All datasets available! Proceeding with preparation...")
        
        aimo1 = manager.load_aimo1()
        aimo2 = manager.load_aimo2()
        
        if aimo1 is not None:
            manager.validate_dataset(aimo1, "AIMO1")
            manager.get_dataset_statistics(aimo1, "AIMO1")
        
        if aimo2 is not None:
            manager.validate_dataset(aimo2, "AIMO2")
            manager.get_dataset_statistics(aimo2, "AIMO2")
        
        # Combine and prepare
        train_df, val_df, test_df = manager.prepare_for_training()
        
        if train_df is not None:
            print("\n✅ Training data prepared!")
            print(f"   Train: {len(train_df)} problems")
            print(f"   Val: {len(val_df)} problems")
            if test_df is not None:
                print(f"   Test: {len(test_df)} problems")
    else:
        print("\n⚠️ Datasets not yet downloaded.")
        print("Please download from Kaggle competition page (see instructions above).")
    
    print("\n" + "="*80)
    print("💡 NEXT STEPS (Task 3.2)")
    print("="*80)
    print("Once datasets are downloaded:")
    print("  1. Place files in datasets/ directory")
    print("  2. Run this script to validate")
    print("  3. Proceed to Task 3.2: Create training data for fine-tuning")
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_task_3_1()
