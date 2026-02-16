"""
data_preparation.py - Data Loading, Validation, and Synthetic Data Generation

Handles AIMO dataset preparation, LaTeX validation, and synthetic problem generation.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and validate AIMO datasets."""
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load CSV dataset.
        
        Args:
            filename: Name of CSV file (with or without extension)
            
        Returns:
            Loaded DataFrame
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filename}: {str(e)}")
            return pd.DataFrame()
    
    
    def load_aimo1(self) -> pd.DataFrame:
        """Load AIMO1 public dataset."""
        return self.load_csv("aimo1_public.csv")
    
    
    def load_aimo2(self) -> pd.DataFrame:
        """Load AIMO2 public dataset."""
        return self.load_csv("aimo2_public.csv")
    
    
    def load_aimo3_public(self) -> pd.DataFrame:
        """Load AIMO3 public test set."""
        return self.load_csv("aimo3_public.csv")


class DataPreprocessor:
    """Preprocess and validate datasets."""
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_cols: Required column names
            
        Returns:
            True if all required columns present
        """
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True
    
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by removing rows with missing values.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        df_clean = df.dropna()
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing values")
        
        return df_clean
    
    
    @staticmethod
    def create_splits(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits.
        
        Args:
            df: Dataset to split
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
            random_state: Random seed
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        np.random.seed(random_state)
        indices = np.random.permutation(len(df))
        
        train_idx = int(len(df) * train_ratio)
        val_idx = int(len(df) * (train_ratio + val_ratio))
        
        train = df.iloc[indices[:train_idx]]
        val = df.iloc[indices[train_idx:val_idx]]
        test = df.iloc[indices[val_idx:]]
        
        logger.info(f"Created splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
    
    
    @staticmethod
    def save_splits(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        output_dir: str = "datasets"
    ):
        """
        Save train/val/test splits to CSV.
        
        Args:
            train: Training set
            val: Validation set
            test: Test set
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        train.to_csv(f"{output_dir}/train_split.csv", index=False)
        val.to_csv(f"{output_dir}/val_split.csv", index=False)
        test.to_csv(f"{output_dir}/test_split.csv", index=False)
        
        logger.info(f"Saved splits to {output_dir}")


class SyntheticDataGenerator:
    """Generate synthetic math problems for testing and training."""
    
    @staticmethod
    def generate_arithmetic_problems(count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate simple arithmetic problems.
        
        Args:
            count: Number of problems to generate
            
        Returns:
            List of problem dictionaries
        """
        problems = []
        
        for i in range(count):
            a = np.random.randint(1, 100)
            b = np.random.randint(1, 100)
            op = np.random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = a + b
                problem = f"Compute {a} + {b}"
            elif op == '-':
                answer = a - b
                problem = f"Compute {a} - {b}"
            else:
                answer = a * b
                problem = f"Compute {a} × {b}"
            
            problems.append({
                "problem_id": f"SYNTH_ARITH_{i:04d}",
                "problem": problem,
                "latex_problem": f"${a} {op} {b}$",
                "answer": answer,
                "category": "arithmetic"
            })
        
        return problems
    
    
    @staticmethod
    def generate_algebra_problems(count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate algebra problems.
        
        Args:
            count: Number of problems to generate
            
        Returns:
            List of problem dictionaries
        """
        problems = []
        problem_idx = 0
        
        while len(problems) < count:
            a = np.random.randint(1, 10)
            # Ensure b and answer are such that c is positive
            answer = np.random.randint(1, 20)
            b = np.random.randint(1, 20)
            c = a * answer + b  # Ensure exact solution
            
            problem = f"Solve {a}x + {b} = {c}. What is x?"
            latex_problem = f"${a}x + {b} = {c}$"
            
            problems.append({
                "problem_id": f"SYNTH_ALG_{problem_idx:04d}",
                "problem": problem,
                "latex_problem": latex_problem,
                "answer": answer,
                "category": "algebra"
            })
            problem_idx += 1
        
        return problems
    
    
    @staticmethod
    def generate_combinatorics_problems(count: int = 10) -> List[Dict[str, Any]]:
        """
        Generate combinatorics problems.
        
        Args:
            count: Number of problems to generate
            
        Returns:
            List of problem dictionaries
        """
        problems = []
        
        for i in range(count):
            n = np.random.randint(3, 10)
            k = np.random.randint(1, n)
            
            # C(n, k) = n! / (k! * (n-k)!)
            from math import comb
            answer = comb(n, k)
            
            problem = f"How many ways are there to choose {k} items from {n} items?"
            latex_problem = f"$\\binom{{{n}}}{{{k}}}$"
            
            problems.append({
                "problem_id": f"SYNTH_COMB_{i:04d}",
                "problem": problem,
                "latex_problem": latex_problem,
                "answer": answer,
                "category": "combinatorics"
            })
        
        return problems
    
    
    @staticmethod
    def generate_all_synthetic(
        count_per_category: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic problems across all categories.
        
        Args:
            count_per_category: Number of problems per category
            
        Returns:
            Combined list of synthetic problems
        """
        problems = []
        problems.extend(SyntheticDataGenerator.generate_arithmetic_problems(count_per_category))
        problems.extend(SyntheticDataGenerator.generate_algebra_problems(count_per_category))
        problems.extend(SyntheticDataGenerator.generate_combinatorics_problems(count_per_category))
        
        logger.info(f"Generated {len(problems)} synthetic problems")
        return problems
    
    
    @staticmethod
    def save_synthetic(problems: List[Dict], output_file: str = "datasets/synthetic_problems.csv"):
        """
        Save synthetic problems to CSV.
        
        Args:
            problems: List of problem dictionaries
            output_file: Output file path
        """
        df = pd.DataFrame(problems)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(problems)} synthetic problems to {output_file}")


class LaTeXValidator:
    """Validate and analyze LaTeX expressions in datasets."""
    
    @staticmethod
    def validate_latex(latex_str: str) -> bool:
        """
        Basic LaTeX validation.
        
        Args:
            latex_str: LaTeX string to validate
            
        Returns:
            True if valid LaTeX
        """
        if not isinstance(latex_str, str):
            return False
        
        # Check for balanced braces
        if latex_str.count('{') != latex_str.count('}'):
            return False
        
        # Check for balanced brackets
        if latex_str.count('[') != latex_str.count(']'):
            return False
        
        return True
    
    
    @staticmethod
    def analyze_latex_commands(df: pd.DataFrame, latex_column: str = "latex_problem") -> Dict[str, int]:
        """
        Analyze LaTeX commands used in dataset.
        
        Args:
            df: Dataset
            latex_column: Name of LaTeX column
            
        Returns:
            Dictionary of command frequencies
        """
        import re
        
        commands = {}
        
        for latex_str in df[latex_column]:
            if not isinstance(latex_str, str):
                continue
            
            # Find all LaTeX commands (patterns like \command)
            found_commands = re.findall(r'\\[a-zA-Z]+', latex_str)
            
            for cmd in found_commands:
                commands[cmd] = commands.get(cmd, 0) + 1
        
        # Sort by frequency
        sorted_commands = dict(sorted(commands.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Found {len(sorted_commands)} unique LaTeX commands")
        return sorted_commands


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    print("Generating synthetic problems...")
    problems = SyntheticDataGenerator.generate_all_synthetic(count_per_category=3)
    print(f"Generated {len(problems)} problems")
    print(f"Sample: {problems[0]}")
    
    # Save synthetic data
    SyntheticDataGenerator.save_synthetic(problems)
    
    # Load and validate
    print("\nLoading datasets...")
    loader = DatasetLoader()
    aimo3 = loader.load_aimo3_public()
    
    if not aimo3.empty:
        print(f"Loaded {len(aimo3)} problems from AIMO3")
        
        # Validate LaTeX
        if 'latex_problem' in aimo3.columns:
            validator = LaTeXValidator()
            commands = validator.analyze_latex_commands(aimo3)
            print(f"Top LaTeX commands: {dict(list(commands.items())[:5])}")
        
        # Create splits
        train, val, test = DataPreprocessor.create_splits(aimo3)
        print(f"Created splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
