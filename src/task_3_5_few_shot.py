"""
Phase 3.5: Advanced Chain-of-Thought with Few-Shot Examples
Implements few-shot learning and prompt optimization for better reasoning
"""

import json
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class FewShotExample:
    """Represents a few-shot example"""
    problem: str
    reasoning: str
    answer: str
    difficulty: str = "medium"
    category: str = "algebra"


class FewShotExampleBase:
    """Pre-curated few-shot examples for different problem categories"""
    
    # Algebra examples
    ALGEBRA_EXAMPLES = [
        FewShotExample(
            problem="Solve for x: 3x + 5 = 14",
            reasoning="We need to isolate x. First, subtract 5 from both sides: 3x = 9. Then divide by 3: x = 3.",
            answer="3",
            difficulty="easy",
            category="algebra"
        ),
        FewShotExample(
            problem="Solve for x: 2(x + 3) = 16",
            reasoning="Expand: 2x + 6 = 16. Subtract 6: 2x = 10. Divide by 2: x = 5.",
            answer="5",
            difficulty="easy",
            category="algebra"
        ),
    ]
    
    # Arithmetic examples
    ARITHMETIC_EXAMPLES = [
        FewShotExample(
            problem="What is 12 × 8?",
            reasoning="Multiply 12 by 8: 12 × 8 = 96.",
            answer="96",
            difficulty="easy",
            category="arithmetic"
        ),
    ]
    
    # Combinatorics examples
    COMBINATORICS_EXAMPLES = [
        FewShotExample(
            problem="How many ways can we choose 2 items from 4?",
            reasoning="This is a combination problem: C(4,2) = 4!/(2!×2!) = 24/4 = 6.",
            answer="6",
            difficulty="medium",
            category="combinatorics"
        ),
    ]
    
    # Geometry examples
    GEOMETRY_EXAMPLES = [
        FewShotExample(
            problem="What is the area of a triangle with base 10 and height 6?",
            reasoning="Area = (1/2) × base × height = (1/2) × 10 × 6 = 30.",
            answer="30",
            difficulty="easy",
            category="geometry"
        ),
    ]
    
    @classmethod
    def get_examples_by_category(cls, category: str, num_examples: int = 2) -> List[FewShotExample]:
        """Get few-shot examples by category"""
        examples_map = {
            "algebra": cls.ALGEBRA_EXAMPLES,
            "arithmetic": cls.ARITHMETIC_EXAMPLES,
            "combinatorics": cls.COMBINATORICS_EXAMPLES,
            "geometry": cls.GEOMETRY_EXAMPLES,
        }
        examples = examples_map.get(category.lower(), cls.ALGEBRA_EXAMPLES)
        return examples[:num_examples]


class PromptOptimizer:
    """Optimizes prompts for better LLM reasoning"""
    
    # Prompt templates with varying levels of detail
    TEMPLATES = {
        "basic": """Solve this math problem:

{problem}

Answer: """,
        
        "step_by_step": """Solve this math problem step by step.

Problem: {problem}

Let's think through this:
1. What is being asked?
2. What information do we have?
3. What approach should we use?
4. Let's calculate step by step
5. What is the final answer?

Answer: """,
        
        "structured": """Problem: {problem}

Solution approach:
- Identify key information
- Choose appropriate method
- Work through calculations
- Verify the answer

Step-by-step solution:

Answer: """,
        
        "detailed_reasoning": """Problem: {problem}

Let me solve this carefully:

1. Understanding the problem:
   - What are we asked to find?
   - What constraints do we have?

2. Strategy:
   - What mathematical concept applies?
   - What method should we use?

3. Execution:
   - Let's work through the solution step by step
   - Show all calculations

4. Verification:
   - Does the answer make sense?
   - Can we verify it?

Final answer: """,
        
        "few_shot": """{examples}

Now solve this problem using the same approach:

Problem: {problem}

Following the same reasoning pattern:

Answer: """,
        
        "expert": """You are an expert mathematician solving olympiad-level problems.

Problem: {problem}

Approach this like an olympiad solver:
- Look for elegant solutions
- Consider special cases
- Use key mathematical insights
- Work systematically

Solution:

Answer: """,
    }
    
    def __init__(self, template_type: str = "step_by_step"):
        """Initialize with template type"""
        self.template_type = template_type
        self.template = self.TEMPLATES.get(template_type, self.TEMPLATES["step_by_step"])
    
    def create_prompt(
        self,
        problem: str,
        category: Optional[str] = None,
        include_few_shot: bool = False,
        num_examples: int = 2
    ) -> str:
        """Create optimized prompt"""
        
        if include_few_shot and category and self.template_type == "few_shot":
            examples = FewShotExampleBase.get_examples_by_category(category, num_examples)
            examples_text = self._format_examples(examples)
            return self.template.format(examples=examples_text, problem=problem)
        
        return self.template.format(problem=problem)
    
    @staticmethod
    def _format_examples(examples: List[FewShotExample]) -> str:
        """Format few-shot examples for inclusion in prompt"""
        formatted = "Here are some example problems and their solutions:\n\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Problem: {ex.problem}\n"
            formatted += f"Solution: {ex.reasoning}\n"
            formatted += f"Answer: {ex.answer}\n\n"
        return formatted


class ChainOfThoughtOptimizer:
    """Optimizes chain-of-thought reasoning with multiple strategies"""
    
    # Different reasoning strategies
    STRATEGIES = {
        "linear": "Solve the problem step by step in order",
        "tree_search": "Consider multiple approaches and choose the best one",
        "decompose": "Break the problem into smaller subproblems",
        "analogy": "Compare to similar problems we've seen",
        "verify": "Solve it twice using different methods to verify",
        "extreme": "Check boundary cases and extreme values",
    }
    
    @staticmethod
    def linear_reasoning(problem: str) -> str:
        """Linear step-by-step reasoning"""
        return f"""Solve this step-by-step:

Problem: {problem}

Step 1: Understand what's asked
Step 2: Identify given information
Step 3: Recall relevant formulas/concepts
Step 4: Execute the solution
Step 5: State the final answer

Answer:"""
    
    @staticmethod
    def tree_search_reasoning(problem: str) -> str:
        """Consider multiple approaches"""
        return f"""Solve by considering multiple approaches:

Problem: {problem}

Approach 1: [Direct method]
Approach 2: [Alternative method]
Approach 3: [Verification method]

Which approach is most efficient? Why?
Let's use the best approach and solve:

Answer:"""
    
    @staticmethod
    def decompose_reasoning(problem: str) -> str:
        """Break problem into subproblems"""
        return f"""Decompose this problem into subproblems:

Problem: {problem}

Subproblem 1: [Part 1]
  Solution: ...
Subproblem 2: [Part 2]
  Solution: ...
Subproblem 3: [Combine results]
  Solution: ...

Combining subproblems:

Answer:"""
    
    @staticmethod
    def verify_reasoning(problem: str) -> str:
        """Solve and verify"""
        return f"""Solve and verify:

Problem: {problem}

Method 1: [First approach]
Result: ...

Method 2: [Second approach]
Result: ...

Do both methods agree? Yes/No

Final answer:"""
    
    @staticmethod
    def extreme_case_reasoning(problem: str) -> str:
        """Check extreme cases"""
        return f"""Check extreme cases:

Problem: {problem}

Edge case 1 (minimum values): ...
Edge case 2 (maximum values): ...
Edge case 3 (boundary condition): ...

General solution: ...

Answer:"""


class AdvancedInferencer:
    """Enhanced inference with few-shot learning and prompt optimization"""
    
    def __init__(self, model, tokenizer, device: str = "cpu"):
        """Initialize with model and tokenizer"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_optimizer = PromptOptimizer()
        self.cot_optimizer = ChainOfThoughtOptimizer()
    
    def infer_with_few_shot(
        self,
        problem: str,
        category: Optional[str] = None,
        template: str = "step_by_step",
        use_few_shot: bool = True,
        num_examples: int = 2,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, any]:
        """
        Infer with few-shot learning and prompt optimization
        
        Args:
            problem: Math problem to solve
            category: Problem category for few-shot selection
            template: Prompt template type
            use_few_shot: Whether to include few-shot examples
            num_examples: Number of few-shot examples
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            Dict with answer, reasoning, confidence
        """
        
        # Create optimized prompt
        self.prompt_optimizer.template_type = template
        if use_few_shot:
            prompt = self.prompt_optimizer.create_prompt(
                problem,
                category=category,
                include_few_shot=True,
                num_examples=num_examples
            )
        else:
            prompt = self.prompt_optimizer.create_prompt(problem)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate
        try:
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning = text.split("Answer:")[-1].strip()
            
            # Extract answer
            answer = self._extract_answer(reasoning)
            confidence = self._estimate_confidence(reasoning, answer)
            
            return {
                "problem": problem,
                "reasoning": reasoning,
                "answer": answer,
                "confidence": confidence,
                "prompt_template": template,
                "category": category,
                "few_shot_used": use_few_shot
            }
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "problem": problem,
                "reasoning": f"Error: {str(e)}",
                "answer": "0",
                "confidence": 0.0,
                "error": True
            }
    
    def multi_prompt_ensemble(
        self,
        problem: str,
        category: Optional[str] = None,
        templates: Optional[List[str]] = None,
        voting: str = "majority"
    ) -> Dict[str, any]:
        """
        Generate predictions using multiple prompt templates
        Then ensemble the results
        
        Args:
            problem: Math problem
            category: Problem category
            templates: List of template types to use
            voting: Voting method (majority, confidence, all)
            
        Returns:
            Ensembled result
        """
        
        if templates is None:
            templates = ["step_by_step", "structured", "detailed_reasoning"]
        
        results = []
        for template in templates:
            result = self.infer_with_few_shot(
                problem,
                category=category,
                template=template,
                use_few_shot=True
            )
            results.append(result)
        
        # Ensemble results
        answers = [r["answer"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        if voting == "majority":
            final_answer = max(set(answers), key=answers.count) if answers else "0"
        elif voting == "confidence":
            final_answer = results[confidences.index(max(confidences))]["answer"]
        else:
            final_answer = answers[0]
        
        return {
            "problem": problem,
            "final_answer": final_answer,
            "candidate_answers": list(set(answers)),
            "all_results": results,
            "agreement": len(set(answers)) == 1,  # All agree?
            "voting_method": voting,
            "confidence": max(confidences) if confidences else 0.0
        }
    
    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract numeric answer from text"""
        # Try "Answer: <num>"
        match = re.search(r'(?:Answer|Final)[:\s]+(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Find last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else "0"
    
    @staticmethod
    def _estimate_confidence(reasoning: str, answer: str) -> float:
        """Estimate confidence in answer"""
        confidence = 0.5
        
        # Check for explicit confidence markers
        if any(word in reasoning.lower() for word in ["clearly", "definitely", "certainly"]):
            confidence += 0.2
        
        # Check for verification language
        if any(word in reasoning.lower() for word in ["verify", "confirm", "check"]):
            confidence += 0.15
        
        # Check for reasoning length (more reasoning = more confident)
        words = len(reasoning.split())
        if words > 50:
            confidence += 0.15
        
        return min(confidence, 1.0)


# Example usage
if __name__ == "__main__":
    logger.info("Phase 3.5: Few-shot learning and prompt optimization")
    
    # Show available templates
    logger.info("\nAvailable prompt templates:")
    for name in PromptOptimizer.TEMPLATES.keys():
        logger.info(f"  - {name}")
    
    # Show reasoning strategies
    logger.info("\nAvailable reasoning strategies:")
    for name, desc in ChainOfThoughtOptimizer.STRATEGIES.items():
        logger.info(f"  - {name}: {desc}")
    
    # Show example few-shot
    logger.info("\nExample few-shot examples (Algebra):")
    examples = FewShotExampleBase.get_examples_by_category("algebra", 2)
    for ex in examples:
        logger.info(f"  Problem: {ex.problem}")
        logger.info(f"  Answer: {ex.answer}\n")
    
    logger.info("✅ Phase 3.5 components ready for integration")
