"""
reasoning.py - LLM-based Mathematical Reasoning Module

Implements chain-of-thought reasoning using open-source LLMs
to solve mathematical problems step-by-step.
"""

import re
import logging
from typing import Optional, Dict, Any
import torch

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers torch")


logger = logging.getLogger(__name__)


class LLMSolver:
    """
    LLM-based solver for mathematical problems using chain-of-thought reasoning.
    """
    
    def __init__(self, model_name: str = "Open-Orca/orca_mini_3b", device: str = "auto"):
        """
        Initialize the LLM solver.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or "auto")
        """
        self.model_name = model_name
        self.device = device
        
        try:
            logger.info(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info(f"Loading model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device == "auto" else None,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32
            )
            
            if device != "auto" and device != "cpu":
                self.model = self.model.to(device)
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    
    def solve(
        self,
        problem_text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Solve a mathematical problem using the LLM.
        
        Args:
            problem_text: The problem to solve
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary with 'reasoning' and 'raw_output' keys
        """
        prompt = self._create_prompt(problem_text)
        
        try:
            logger.info("Generating reasoning...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning = self._extract_reasoning(raw_output)
            
            return {
                "problem": problem_text,
                "reasoning": reasoning,
                "raw_output": raw_output,
                "prompt": prompt
            }
        except Exception as e:
            logger.error(f"Error during solving: {str(e)}")
            raise
    
    
    def _create_prompt(self, problem_text: str) -> str:
        """
        Create a chain-of-thought prompt for the problem.
        
        Args:
            problem_text: The problem to solve
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a mathematical expert solving Olympiad-level problems.
Solve the following problem step-by-step:

Problem: {problem_text}

Solution:
Let me work through this carefully.

Step 1: """
        return prompt
    
    
    def _extract_reasoning(self, raw_output: str) -> str:
        """
        Extract the reasoning portion from the model output.
        
        Args:
            raw_output: Full model output
            
        Returns:
            Extracted reasoning text
        """
        # Find the solution part after "Solution:" or "Let me work"
        if "Solution:" in raw_output:
            reasoning = raw_output.split("Solution:")[-1]
        elif "Step" in raw_output:
            reasoning = raw_output.split("Step 1:")[-1]
        else:
            reasoning = raw_output
        
        return reasoning.strip()
    
    
    def batch_solve(
        self,
        problems: list,
        max_tokens: int = 512,
        verbose: bool = True
    ) -> list:
        """
        Solve multiple problems.
        
        Args:
            problems: List of problem texts
            max_tokens: Maximum tokens per problem
            verbose: Whether to log progress
            
        Returns:
            List of solution dictionaries
        """
        results = []
        total = len(problems)
        
        for idx, problem in enumerate(problems):
            if verbose:
                logger.info(f"Solving problem {idx + 1}/{total}...")
            
            try:
                result = self.solve(problem, max_tokens=max_tokens)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to solve problem {idx + 1}: {str(e)}")
                results.append({
                    "problem": problem,
                    "reasoning": None,
                    "raw_output": None,
                    "error": str(e)
                })
        
        return results


def extract_numeric_answer(text: str) -> Optional[int]:
    """
    Extract a numeric answer from text output.
    
    Args:
        text: Text to extract answer from
        
    Returns:
        Extracted integer or None
    """
    # Look for patterns like "Answer: 123" or "Final answer: 123"
    patterns = [
        r"(?:answer|result|final answer)\s*:?\s*(\d+)",
        r"(?:the answer is|equals)\s*(\d+)",
        r"(\d{1,5})\s*(?:is the answer|is correct)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    # Fallback: find all numbers and return the largest one (usually the final answer)
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    
    return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    solver = LLMSolver(model_name="Open-Orca/orca_mini_3b")
    
    problem = "Compute 2 + 3 × 5"
    result = solver.solve(problem)
    
    print("Problem:", result["problem"])
    print("\nReasoning:")
    print(result["reasoning"])
    
    answer = extract_numeric_answer(result["reasoning"])
    print(f"\nExtracted answer: {answer}")
