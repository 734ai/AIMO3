"""
Process MATH dataset (ai|mo-dataset00.csv) and split into AIMO-compatible datasets.

This script:
1. Reads the full MATH dataset (146K+ problems)
2. Extracts relevant columns (problem, answer)
3. Splits into AIMO1 and AIMO2 based on difficulty levels
4. Generates CSV files compatible with fine-tuning pipeline
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MATHDatasetProcessor:
    """Process MATH dataset and create AIMO-compatible splits."""
    
    def __init__(self, dataset_path: str = "datasets/ai|mo-dataset00.csv"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path("datasets")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the MATH dataset."""
        print("\n" + "="*80)
        print("📂 LOADING MATH DATASET")
        print("="*80)
        
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found: {self.dataset_path}")
            return None
        
        # Read CSV with proper handling of quoted fields
        df = pd.read_csv(self.dataset_path)
        print(f"\n✅ Dataset loaded: {len(df)} problems")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def extract_answer(self, solution_text: str) -> str:
        """Extract the final answer from solution text."""
        if pd.isna(solution_text):
            return None
        
        solution_text = str(solution_text)
        
        # Look for \boxed{...} patterns (standard math notation)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
        if boxed_match:
            answer = boxed_match.group(1)
            # Clean up LaTeX
            answer = re.sub(r'\\[a-z]+\{', '', answer)  # Remove LaTeX commands
            answer = re.sub(r'[{}]', '', answer)  # Remove braces
            answer = answer.strip()
            if answer:
                return answer
        
        # Fallback: try to extract last numeric value
        numbers = re.findall(r'-?\d+(?:\.\d+)?', solution_text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def classify_difficulty(self, level_str: str) -> int:
        """Convert level string to numeric difficulty (1-5)."""
        if pd.isna(level_str):
            return 3  # Default to medium
        
        level_str = str(level_str).lower().strip()
        
        if 'level 1' in level_str or 'level1' in level_str:
            return 1
        elif 'level 2' in level_str or 'level2' in level_str:
            return 2
        elif 'level 3' in level_str or 'level3' in level_str:
            return 3
        elif 'level 4' in level_str or 'level4' in level_str:
            return 4
        elif 'level 5' in level_str or 'level5' in level_str:
            return 5
        
        return 3  # Default
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataset: extract answers, clean data, add difficulty."""
        print("\n" + "="*80)
        print("🔧 PROCESSING DATASET")
        print("="*80)
        
        # Extract answers from solutions
        print("\n📝 Extracting answers from solutions...")
        df['answer'] = df['solution'].apply(self.extract_answer)
        
        # Extract numeric difficulty
        print("   ✓ Extracting difficulty levels...")
        df['difficulty'] = df['level'].apply(self.classify_difficulty)
        
        # Keep only rows with valid answers
        print("   ✓ Filtering valid answers...")
        valid_rows = df[df['answer'].notna()].copy()
        print(f"\n✅ Processing complete: {len(valid_rows)}/{len(df)} rows with valid answers")
        
        # Show statistics
        print(f"\n📊 Difficulty Distribution:")
        for level in range(1, 6):
            count = (valid_rows['difficulty'] == level).sum()
            print(f"   Level {level}: {count:6d} problems ({count/len(valid_rows)*100:5.1f}%)")
        
        return valid_rows
    
    def split_datasets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into AIMO1 and AIMO2 based on difficulty and size."""
        print("\n" + "="*80)
        print("✂️ SPLITTING INTO AIMO1 AND AIMO2")
        print("="*80)
        
        # AIMO1: Easier problems (Levels 1-2) - focus on foundation
        # AIMO2: Mixed difficulty (Levels 2-5) - comprehensive training
        
        aimo1_mask = df['difficulty'] <= 2
        aimo1 = df[aimo1_mask].copy()
        
        # AIMO2: Take remaining + some from level 3
        aimo2_mask = df['difficulty'] >= 2
        aimo2 = df[aimo2_mask].copy()
        
        print(f"\n✅ Split complete:")
        print(f"   AIMO1 (Levels 1-2):  {len(aimo1):6d} problems")
        print(f"   AIMO2 (Levels 2-5):  {len(aimo2):6d} problems")
        
        return aimo1, aimo2
    
    def prepare_aimo_format(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Prepare dataframe in AIMO format."""
        result = pd.DataFrame({
            'problem': df['problem'],
            'answer': df['answer'],
            'difficulty': df['difficulty'],
            'category': df.get('type', 'Algebra'),  # Type is closest to category
            'solution': df['solution'],
            'stage': df.get('stage', 'train'),
            'source': dataset_name
        })
        
        return result
    
    def save_datasets(self, aimo1: pd.DataFrame, aimo2: pd.DataFrame):
        """Save datasets to CSV files."""
        print("\n" + "="*80)
        print("💾 SAVING DATASETS")
        print("="*80)
        
        # Save AIMO1
        aimo1_path = self.output_dir / "aimo1_problems.csv"
        aimo1_prepared = self.prepare_aimo_format(aimo1, "MATH")
        aimo1_prepared.to_csv(aimo1_path, index=False)
        print(f"\n✅ AIMO1 saved: {aimo1_path}")
        print(f"   Rows: {len(aimo1_prepared)}")
        print(f"   Size: {aimo1_path.stat().st_size / 1024**2:.2f} MB")
        print(f"   Columns: {list(aimo1_prepared.columns)}")
        
        # Save AIMO2
        aimo2_path = self.output_dir / "aimo2_problems.csv"
        aimo2_prepared = self.prepare_aimo_format(aimo2, "MATH")
        aimo2_prepared.to_csv(aimo2_path, index=False)
        print(f"\n✅ AIMO2 saved: {aimo2_path}")
        print(f"   Rows: {len(aimo2_prepared)}")
        print(f"   Size: {aimo2_path.stat().st_size / 1024**2:.2f} MB")
        print(f"   Columns: {list(aimo2_prepared.columns)}")
        
        return aimo1_prepared, aimo2_prepared
    
    def generate_statistics(self, aimo1: pd.DataFrame, aimo2: pd.DataFrame):
        """Generate and display dataset statistics."""
        print("\n" + "="*80)
        print("📊 DATASET STATISTICS")
        print("="*80)
        
        print("\n📈 AIMO1 Statistics:")
        print(f"   Total problems: {len(aimo1)}")
        print(f"   Problem length (avg): {aimo1['problem'].str.len().mean():.0f} chars")
        print(f"   Answer length (avg): {aimo1['answer'].str.len().mean():.0f} chars")
        print(f"   Difficulty distribution:")
        for level in range(1, 6):
            count = (aimo1['difficulty'] == level).sum()
            if count > 0:
                print(f"      Level {level}: {count:6d}")
        
        print(f"\n📈 AIMO2 Statistics:")
        print(f"   Total problems: {len(aimo2)}")
        print(f"   Problem length (avg): {aimo2['problem'].str.len().mean():.0f} chars")
        print(f"   Answer length (avg): {aimo2['answer'].str.len().mean():.0f} chars")
        print(f"   Difficulty distribution:")
        for level in range(1, 6):
            count = (aimo2['difficulty'] == level).sum()
            if count > 0:
                print(f"      Level {level}: {count:6d}")
        
        print(f"\n📚 Category Distribution (AIMO1):")
        categories = aimo1['category'].value_counts()
        for cat, count in categories.head(5).items():
            print(f"   {cat}: {count:6d}")
        
        print(f"\n📚 Category Distribution (AIMO2):")
        categories = aimo2['category'].value_counts()
        for cat, count in categories.head(5).items():
            print(f"   {cat}: {count:6d}")
    
    def process(self):
        """Main processing pipeline."""
        print("\n" + "="*80)
        print("🚀 PROCESSING MATH DATASET FOR AIMO")
        print("="*80)
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            return
        
        # Process dataset
        df_processed = self.process_dataset(df)
        
        # Split into AIMO1 and AIMO2
        aimo1, aimo2 = self.split_datasets(df_processed)
        
        # Save datasets
        aimo1_saved, aimo2_saved = self.save_datasets(aimo1, aimo2)
        
        # Generate statistics
        self.generate_statistics(aimo1_saved, aimo2_saved)
        
        print("\n" + "="*80)
        print("✅ PROCESSING COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("   1. Verify datasets: python src/task_3_1_datasets.py")
        print("   2. Generate training data: python src/task_3_2_3_fine_tuning.py")
        print("   3. Start fine-tuning: python src/task_3_2_3_fine_tuning.py")


if __name__ == "__main__":
    processor = MATHDatasetProcessor()
    processor.process()
