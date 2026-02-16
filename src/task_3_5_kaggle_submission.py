"""
AIMO3 Kaggle Submission Generation
Generates predictions for AIMO3 test set and outputs in submission format
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SubmissionGenerator:
    """Generate AIMO3 competition submissions"""
    
    def __init__(self, model_path: str = "gpt2", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize submission generator
        
        Args:
            model_path: Path to fine-tuned model or model name
            device: torch device (cuda/cpu)
        """
        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_model(self) -> None:
        """Load tokenizer and model"""
        logger.info(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_prompt(self, problem: str, include_cot: bool = True) -> str:
        """Create prompt for problem-solving"""
        if include_cot:
            return f"""Solve this mathematical problem step by step.

Problem: {problem}

Let's work through this carefully:
1. First, let's understand what we're asked to find
2. Then, let's solve it step by step
3. Finally, let's extract the numerical answer

Answer:"""
        else:
            return f"""Problem: {problem}

Answer:"""
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract numeric answer from generated text
        
        Args:
            text: Generated text from model
            
        Returns:
            Extracted answer string or None
        """
        import re
        
        text = text.strip()
        
        # Try to find "Answer: <number>"
        match = re.search(r'(?:Answer|Final Answer|The answer is)[:\s]+(-?\d+(?:\.?\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Try to find last number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def generate_answer(self, problem: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """
        Generate answer for a single problem
        
        Args:
            problem: Problem statement
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        if self.model is None:
            self.load_model()
        
        prompt = self.create_prompt(problem, include_cot=True)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        answer = self.extract_answer(generated_text.split("Answer:")[-1])
        
        return answer if answer else "0"
    
    def load_test_data(self, test_path: str = "datasets/aimo3_test.csv") -> pd.DataFrame:
        """Load test data"""
        logger.info(f"Loading test data from {test_path}")
        df = pd.read_csv(test_path)
        logger.info(f"Loaded {len(df)} test problems")
        return df
    
    def generate_submission(self, test_path: str = "datasets/aimo3_test.csv") -> pd.DataFrame:
        """
        Generate submission dataframe
        
        Args:
            test_path: Path to test CSV
            
        Returns:
            DataFrame with id and answer columns
        """
        # Load test data
        test_df = self.load_test_data(test_path)
        
        # Generate answers
        logger.info("Generating predictions...")
        answers = []
        
        for idx, row in test_df.iterrows():
            problem_id = row['id']
            problem = row['problem']
            
            logger.info(f"[{idx+1}/{len(test_df)}] Processing {problem_id}")
            
            try:
                answer = self.generate_answer(problem)
                answers.append({
                    'id': problem_id,
                    'answer': answer
                })
                logger.info(f"  → Answer: {answer}")
            except Exception as e:
                logger.error(f"Error processing {problem_id}: {e}")
                answers.append({
                    'id': problem_id,
                    'answer': '0'  # Default to 0 on error
                })
        
        submission_df = pd.DataFrame(answers)
        logger.info(f"Generated {len(submission_df)} predictions")
        
        return submission_df
    
    def save_submission(self, submission_df: pd.DataFrame, output_path: str = "outputs/submission.csv") -> str:
        """
        Save submission to CSV
        
        Args:
            submission_df: DataFrame with predictions
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        
        # Log summary
        logger.info(f"Submission format:")
        logger.info(submission_df.head())
        
        return output_path
    
    def run(self, test_path: str = "datasets/aimo3_test.csv", 
            output_path: str = "outputs/submission.csv",
            model_path: Optional[str] = None) -> str:
        """
        Run complete submission pipeline
        
        Args:
            test_path: Path to test CSV
            output_path: Path to save submission
            model_path: Optional path to fine-tuned model
            
        Returns:
            Path to saved submission file
        """
        if model_path:
            self.model_path = model_path
        
        # Load model
        self.load_model()
        
        # Generate predictions
        submission_df = self.generate_submission(test_path)
        
        # Save submission
        output_file = self.save_submission(submission_df, output_path)
        
        return output_file


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AIMO3 submission")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--test", default="datasets/aimo3_test.csv", help="Test data path")
    parser.add_argument("--output", default="outputs/submission.csv", help="Output path")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = SubmissionGenerator(model_path=args.model, device=device)
    submission_path = generator.run(
        test_path=args.test,
        output_path=args.output,
        model_path=args.model
    )
    
    logger.info(f"✅ Submission ready: {submission_path}")


if __name__ == "__main__":
    main()
