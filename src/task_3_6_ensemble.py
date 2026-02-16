"""
Ensemble Inference Engine for AIMO3
Combines multiple models for better accuracy
"""

import logging
import torch
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from multi_model_support import MultiModelLoader, SUPPORTED_MODELS
except ImportError:
    # Fallback if import fails
    MultiModelLoader = None
    SUPPORTED_MODELS = {}

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a single model prediction"""
    model_name: str
    problem_id: str
    problem: str
    generated_text: str
    extracted_answer: str
    confidence: float
    reasoning: str


class EnsembleInferencer:
    """Ensemble inference using multiple models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.loader = MultiModelLoader(device=device)
        self.models: Dict[str, Tuple] = {}  # model_key -> (model, tokenizer)
        
    def load_models(self, model_keys: List[str], 
                   lora_paths: Optional[Dict[str, str]] = None,
                   quantize: bool = False) -> None:
        """
        Load multiple models
        
        Args:
            model_keys: List of model keys from SUPPORTED_MODELS
            lora_paths: Dict mapping model_key -> lora_path (optional)
            quantize: Use 8-bit quantization
        """
        for model_key in model_keys:
            lora_path = lora_paths.get(model_key) if lora_paths else None
            model, tokenizer = self.loader.load_model(model_key, lora_path, quantize)
            self.models[model_key] = (model, tokenizer)
        
        logger.info(f"✅ Loaded {len(self.models)} models for ensemble")
    
    def _create_prompt(self, problem: str, include_cot: bool = True) -> str:
        """Create prompt with optional chain-of-thought"""
        if include_cot:
            return f"""Solve this mathematical olympiad problem step by step.

Problem: {problem}

Think through this carefully:
1. What are we solving for?
2. What approach should we use?
3. Execute the solution step by step
4. Verify the answer

Answer:"""
        else:
            return f"Problem: {problem}\n\nAnswer:"
    
    def _extract_answer(self, text: str) -> Tuple[str, float]:
        """
        Extract numeric answer from text with confidence score
        
        Returns:
            (answer_string, confidence_score)
        """
        text = text.strip()
        
        # Try explicit answer patterns (higher confidence)
        patterns = [
            (r'(?:Answer|Final Answer|The answer is)[:\s]*([-\d.]+)', 0.95),
            (r'(?:Therefore|So|Thus),?\s+(?:the )?answer\s+(?:is\s+)?([-\d.]+)', 0.90),
            (r'result[:\s]*([-\d.]+)', 0.85),
            (r'equals?\s+([-\d.]+)', 0.80),
        ]
        
        for pattern, confidence in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1), confidence
        
        # Extract all numbers
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            # Last number is often the answer
            return numbers[-1], 0.60
        
        return "0", 0.10
    
    def predict_single(self, model_key: str, problem: str, 
                      max_length: int = 1024, temperature: float = 0.7) -> PredictionResult:
        """
        Get prediction from a single model
        
        Args:
            model_key: Which model to use
            problem: Problem statement
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            PredictionResult with extracted answer
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model, tokenizer = self.models[model_key]
        config = SUPPORTED_MODELS[model_key]
        
        prompt = self._create_prompt(problem, include_cot=config.reasoning_capable)
        
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(max_length, config.max_tokens),
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text
        
        # Extract answer
        answer, confidence = self._extract_answer(reasoning)
        
        return PredictionResult(
            model_name=config.name,
            problem_id="",
            problem=problem,
            generated_text=generated_text,
            extracted_answer=answer,
            confidence=confidence,
            reasoning=reasoning[:200]  # Truncate for logging
        )
    
    def predict_ensemble(self, problem: str, ensemble_method: str = "majority_vote",
                        temperature: float = 0.7) -> Dict[str, any]:
        """
        Get ensemble prediction from all loaded models
        
        Args:
            problem: Problem statement
            ensemble_method: 'majority_vote', 'weighted_vote', 'average', 'max_confidence'
            temperature: Sampling temperature
            
        Returns:
            Dict with final answer and individual predictions
        """
        predictions = []
        
        logger.info(f"Running ensemble with {len(self.models)} models...")
        
        for model_key in self.models:
            try:
                result = self.predict_single(model_key, problem, temperature=temperature)
                predictions.append(result)
                logger.debug(f"  {result.model_name}: {result.extracted_answer} (conf: {result.confidence:.2f})")
            except Exception as e:
                logger.error(f"Error in {model_key}: {e}")
        
        # Ensemble voting
        final_answer = self._ensemble_vote(predictions, ensemble_method)
        
        return {
            "final_answer": final_answer,
            "predictions": predictions,
            "method": ensemble_method,
            "num_models": len(predictions),
            "agreement": self._compute_agreement(predictions)
        }
    
    def _ensemble_vote(self, predictions: List[PredictionResult], 
                       method: str = "majority_vote") -> str:
        """Combine predictions using different methods"""
        answers = [p.extracted_answer for p in predictions]
        
        if method == "majority_vote":
            # Most common answer
            counter = Counter(answers)
            return counter.most_common(1)[0][0]
        
        elif method == "weighted_vote":
            # Vote weighted by confidence
            weighted = {}
            for p in predictions:
                answer = p.extracted_answer
                weighted[answer] = weighted.get(answer, 0) + p.confidence
            return max(weighted.items(), key=lambda x: x[1])[0]
        
        elif method == "max_confidence":
            # Answer from most confident model
            return max(predictions, key=lambda p: p.confidence).extracted_answer
        
        elif method == "average":
            # Average numeric answers
            try:
                numeric = [float(a) for a in answers if re.match(r'-?\d+(?:\.\d+)?', a)]
                if numeric:
                    avg = np.mean(numeric)
                    return str(int(avg)) if avg == int(avg) else str(avg)
            except:
                pass
            # Fall back to majority vote
            return Counter(answers).most_common(1)[0][0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _compute_agreement(self, predictions: List[PredictionResult]) -> float:
        """Compute agreement score (0-1) for ensemble predictions"""
        if len(predictions) <= 1:
            return 1.0
        
        answers = [p.extracted_answer for p in predictions]
        most_common = Counter(answers).most_common(1)[0]
        agreement = most_common[1] / len(predictions)
        
        return agreement
    
    def unload_model(self, model_key: str) -> None:
        """Unload a model to free memory"""
        if model_key in self.models:
            del self.models[model_key]
            torch.cuda.empty_cache()
    
    def unload_all(self) -> None:
        """Unload all models"""
        self.models.clear()
        torch.cuda.empty_cache()


class FastEnsemble:
    """Fast ensemble using smaller, faster models"""
    
    FAST_MODELS = ["gpt2", "gemma3-4b", "gemma3-12b"]
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ensemble = EnsembleInferencer(device=device)
        
    def setup(self) -> None:
        """Setup fast ensemble"""
        self.ensemble.load_models(self.FAST_MODELS)
    
    def predict(self, problem: str) -> str:
        """Quick prediction using fast ensemble"""
        result = self.ensemble.predict_ensemble(
            problem,
            ensemble_method="majority_vote",
            temperature=0.5
        )
        return result["final_answer"]


class QualityEnsemble:
    """High-quality ensemble using larger reasoning models"""
    
    QUALITY_MODELS = ["llama4-maverick", "qwen3-72b", "mistral-large-3"]
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ensemble = EnsembleInferencer(device=device)
    
    def setup(self, quantize: bool = True) -> None:
        """Setup quality ensemble (requires quantization for most GPUs)"""
        self.ensemble.load_models(self.QUALITY_MODELS, quantize=quantize)
    
    def predict(self, problem: str) -> str:
        """High-quality prediction using quality ensemble"""
        result = self.ensemble.predict_ensemble(
            problem,
            ensemble_method="weighted_vote",
            temperature=0.7
        )
        return result["final_answer"]


def demo_ensemble():
    """Demo ensemble inference"""
    # Initialize
    ensemble = EnsembleInferencer()
    
    # Load fast models for demo
    ensemble.load_models(["gpt2"])  # Start with just GPT-2 for demo
    
    # Test problems
    problems = [
        "What is 2 + 2?",
        "Solve for x: x + 5 = 10",
        "What is the square root of 144?",
    ]
    
    for problem in problems:
        result = ensemble.predict_ensemble(problem, ensemble_method="majority_vote")
        print(f"\nProblem: {problem}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Agreement: {result['agreement']:.2%}")
        print(f"Predictions:")
        for pred in result['predictions']:
            print(f"  - {pred.model_name}: {pred.extracted_answer} (conf: {pred.confidence:.2f})")


if __name__ == "__main__":
    # Show models
    from src.multi_model_support import compare_models_performance
    compare_models_performance()
    
    # Run demo
    print("\n" + "="*80)
    print("ENSEMBLE INFERENCE DEMO")
    print("="*80)
    demo_ensemble()
