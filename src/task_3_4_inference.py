"""
Task 3.4: Model Inference & Testing
====================================

Test the fine-tuned (or base) model on AIMO problems.
Includes chain-of-thought prompting and answer extraction.
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIOMInferencer:
    """Inference engine for AIMO problem solving."""
    
    def __init__(self, model_path: Optional[str] = None, base_model: str = "gpt2"):
        """
        Initialize inferencer with model.
        
        Args:
            model_path: Path to fine-tuned LoRA weights. If None, uses base model.
            base_model: Base model name (gpt2, mistral-7b, etc.)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else None
        self.base_model = base_model
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print("\n" + "="*80)
        print("🤖 LOADING MODEL FOR INFERENCE")
        print("="*80)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            # Load tokenizer
            print(f"\n📥 Loading tokenizer: {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Tokenizer loaded")
            
            # Load base model
            print(f"📥 Loading base model: {self.base_model}")
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model)
            print(f"✅ Base model loaded ({sum(p.numel() for p in self.model.parameters()):,} parameters)")
            
            # Load LoRA weights if available
            if self.model_path and self.model_path.exists():
                print(f"📥 Loading LoRA weights: {self.model_path}")
                self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
                print(f"✅ LoRA weights loaded")
            else:
                print(f"⚠️ No LoRA weights found, using base model only")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            print(f"\n❌ Model loading failed: {e}")
            raise
    
    def create_prompt(self, problem: str, include_cot: bool = True) -> str:
        """
        Create prompt for problem solving.
        
        Args:
            problem: Mathematical problem statement
            include_cot: Include chain-of-thought instruction
            
        Returns:
            Formatted prompt
        """
        if include_cot:
            prompt = f"""Solve this math problem step by step.

Problem: {problem}

Let me think about this step by step:
1. First, I'll identify what we need to find
2. Then, I'll set up the equations or approach
3. Finally, I'll solve it

Solution:"""
        else:
            prompt = f"""Solve this math problem.

Problem: {problem}

Answer:"""
        
        return prompt
    
    def generate(
        self,
        problem: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        num_beams: int = 1,
        do_sample: bool = True,
        include_cot: bool = True
    ) -> str:
        """
        Generate solution for a problem.
        
        Args:
            problem: Problem statement
            max_length: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Beam search width
            do_sample: Use sampling vs greedy
            include_cot: Include chain-of-thought
            
        Returns:
            Generated solution text
        """
        prompt = self.create_prompt(problem, include_cot)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        answer = generated_text[len(prompt):].strip()
        
        return answer
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract numeric answer from generated text.
        
        Args:
            text: Generated solution text
            
        Returns:
            Extracted answer or None
        """
        # Look for common patterns: "Answer: 42", "Answer is 42", etc.
        patterns = [
            r'[Aa]nswer[:\s]+(\d+)',  # Answer: 42 or Answer 42
            r'=\s*(\d+)',  # = 42
            r'(?:The answer is|Final answer|Result)[:\s]+(\d+)',  # Explicit patterns
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Fallback: extract last number
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def solve(
        self,
        problem: str,
        include_cot: bool = True,
        extract_numeric: bool = True
    ) -> Dict:
        """
        Solve a problem end-to-end.
        
        Args:
            problem: Problem statement
            include_cot: Use chain-of-thought
            extract_numeric: Extract numeric answer
            
        Returns:
            Dict with reasoning and answer
        """
        # Generate
        reasoning = self.generate(problem, include_cot=include_cot)
        
        # Extract answer
        answer = None
        if extract_numeric:
            answer = self.extract_answer(reasoning)
        
        return {
            "problem": problem,
            "reasoning": reasoning,
            "answer": answer,
            "method": "cot" if include_cot else "direct",
            "model": self.base_model,
            "has_lora": self.model_path is not None,
        }


def test_inference():
    """Test inference on sample problems."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*18 + "🧪 TASK 3.4: MODEL INFERENCE & TESTING" + " "*20 + "║")
    print("║" + " "*20 + "Evaluate model on AIMO problems" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    # Load test data
    test_data_path = Path("datasets/training/test.jsonl")
    if not test_data_path.exists():
        print("\n❌ Test data not found!")
        print("   Run: python src/task_3_2_3_fine_tuning.py")
        return
    
    # Load test examples
    test_examples = []
    with open(test_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # Test on first 5 examples
                test_examples.append(json.loads(line))
    
    print(f"\n✅ Loaded {len(test_examples)} test examples")
    
    # Initialize inferencer
    try:
        inferencer = AIOMInferencer(base_model="gpt2")
    except Exception as e:
        print(f"\n❌ Failed to initialize inferencer: {e}")
        return
    
    # Test on examples
    print("\n" + "="*80)
    print("🧪 TESTING INFERENCE")
    print("="*80)
    
    results = []
    for i, example in enumerate(test_examples, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}/{len(test_examples)}")
        print(f"{'─'*80}")
        
        problem = example["problem"]
        expected_answer = example["answer"]
        
        # Shorten long problems for display
        display_problem = problem[:100] + "..." if len(problem) > 100 else problem
        print(f"\n📝 Problem: {display_problem}")
        print(f"✓ Expected Answer: {expected_answer}")
        
        try:
            # Solve
            print(f"\n🤖 Generating solution (using chain-of-thought)...")
            result = inferencer.solve(problem, include_cot=True, extract_numeric=True)
            
            # Show reasoning (first 200 chars)
            reasoning_display = result["reasoning"][:200].replace('\n', ' ')
            print(f"\n💭 Reasoning: {reasoning_display}...")
            
            # Show extracted answer
            if result["answer"]:
                match = "✅" if result["answer"] == expected_answer else "❌"
                print(f"{match} Generated Answer: {result['answer']}")
            else:
                print(f"❌ Could not extract numeric answer")
            
            results.append(result)
        
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            results.append({"error": str(e)})
    
    # Summary
    print("\n" + "="*80)
    print("📊 INFERENCE SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if "error" not in r)
    with_answer = sum(1 for r in results if r.get("answer"))
    correct = sum(1 for r in results if r.get("answer") == r.get("expected_answer"))
    
    print(f"\n✅ Successful inferences: {successful}/{len(results)}")
    print(f"✅ With extracted answers: {with_answer}/{len(results)}")
    
    if with_answer > 0:
        print(f"   Accuracy (on extracted): {correct}/{with_answer}")
    
    # Save results
    results_file = Path("outputs/inference_results.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n📁 Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print("✨ INFERENCE TESTING COMPLETE")
    print("="*80)
    print("\n📋 Next Steps (Task 3.5):")
    print("   • Implement ensemble methods")
    print("   • Combine multiple model predictions")
    print("   • Weighted voting mechanisms")


if __name__ == "__main__":
    test_inference()
