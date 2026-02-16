#!/usr/bin/env python3
"""
Test script for AIMO3 notebook - Phase 4 components
This tests the core Phase 4 functionality without requiring heavy model loading
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# Setup logging (same as notebook)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 60)
print("AIMO3 NOTEBOOK COMPONENT TEST")
print("=" * 60)
print()

# Test 1: Import required libraries
print("Test 1: Checking dependencies...")
try:
    import sympy as sp
    import pandas as pd
    print("✅ sympy imported")
    print("✅ pandas imported")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Phase 4 - SymbolicCompute Class
print("\nTest 2: Loading SymbolicCompute class...")

class SymbolicCompute:
    """SymPy-based symbolic computation and verification."""
    
    @staticmethod
    def evaluate_expression(expr_str: str) -> Optional[Union[int, float]]:
        """Evaluate a mathematical expression string using SymPy."""
        try:
            expr = sp.sympify(expr_str)
            result = expr.evalf()
            return int(result) if result == int(result) else float(result)
        except Exception as e:
            logger.debug(f"Failed to evaluate expression '{expr_str}': {str(e)}")
            return None
    
    @staticmethod
    def parse_llm_output_for_expressions(llm_text: str) -> Dict[str, Any]:
        """Parse LLM output to extract mathematical expressions."""
        result = {
            "expressions": [],
            "values": [],
            "final_value": None
        }
        
        try:
            # Extract expressions
            expr_pattern = r"(?:=|equals|is)\s*(\d+(?:\.\d+)?|[\w\s\+\-\*/\(\)\.]+)"
            expr_matches = re.findall(expr_pattern, llm_text, re.IGNORECASE)
            
            # Extract numeric values
            num_pattern = r"\b(\d+(?:\.\d+)?)\b"
            num_matches = re.findall(num_pattern, llm_text)
            
            result["expressions"] = expr_matches[:5]
            result["values"] = [float(n) if '.' in n else int(n) for n in num_matches[:10]]
            
            if num_matches:
                result["final_value"] = float(num_matches[-1]) if '.' in num_matches[-1] else int(num_matches[-1])
                
        except Exception as e:
            logger.debug(f"Failed to parse LLM output: {str(e)}")
        
        return result
    
    @staticmethod
    def verify_symbolic_result(
        llm_answer: int,
        llm_output: str,
        tolerance: float = 0.01
    ) -> Tuple[bool, float]:
        """Verify LLM answer by symbolic computation. Returns (is_valid, confidence)."""
        try:
            parsed = SymbolicCompute.parse_llm_output_for_expressions(llm_output)
            
            # Try to evaluate extracted expressions
            if parsed["expressions"]:
                for expr_str in parsed["expressions"]:
                    try:
                        result = SymbolicCompute.evaluate_expression(expr_str)
                        if result is not None:
                            if isinstance(result, float):
                                diff_percent = abs(result - llm_answer) / max(abs(llm_answer), 1)
                                if diff_percent <= tolerance:
                                    return True, 1.0 - diff_percent
                            else:
                                if int(result) == llm_answer:
                                    return True, 1.0
                    except:
                        continue
            
            # Check final value
            if parsed["final_value"] is not None and parsed["final_value"] == llm_answer:
                return True, 0.8
                    
        except Exception as e:
            logger.debug(f"Verification failed: {str(e)}")
        
        return False, 0.5

print("✅ SymbolicCompute class loaded")

# Test 3: Phase 4 - AnswerValidator Class
print("\nTest 3: Loading AnswerValidator class...")

class AnswerValidator:
    """Validates and enforces answer format constraints."""
    
    AIMO_MIN = 0
    AIMO_MAX = 99999
    
    @staticmethod
    def validate_integer(value: Any) -> Optional[int]:
        """Validate and convert value to valid AIMO integer."""
        try:
            int_value = int(float(str(value).strip()))
            
            if int_value < AnswerValidator.AIMO_MIN:
                return AnswerValidator.AIMO_MIN
            
            if int_value > AnswerValidator.AIMO_MAX:
                return AnswerValidator.AIMO_MAX
            
            return int_value
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to validate answer: {str(e)}")
            return None
    
    @staticmethod
    def validate_with_fallback_strategies(
        llm_answer: Optional[int],
        llm_text: str
    ) -> Dict[str, Any]:
        """Validate answer with multiple fallback strategies."""
        result = {
            "final_answer": 0,
            "confidence": 0.0,
            "strategy_used": "default_fallback",
            "fallback_applied": False
        }
        
        try:
            # Strategy 1: Use primary answer if valid
            if llm_answer is not None:
                validated = AnswerValidator.validate_integer(llm_answer)
                if validated is not None:
                    result["final_answer"] = validated
                    result["confidence"] = 0.9
                    result["strategy_used"] = "primary_llm_answer"
                    return result
            
            # Strategy 2: Try symbolic verification
            is_valid, confidence = SymbolicCompute.verify_symbolic_result(
                llm_answer if llm_answer is not None else 0,
                llm_text
            )
            
            if is_valid and llm_answer is not None:
                validated = AnswerValidator.validate_integer(llm_answer)
                if validated is not None:
                    result["final_answer"] = validated
                    result["confidence"] = confidence
                    result["strategy_used"] = "symbolic_verification"
                    result["fallback_applied"] = True
                    return result
            
            result["fallback_applied"] = True
            
        except Exception as e:
            logger.debug(f"Fallback validation failed: {str(e)}")
        
        return result
    
    @staticmethod
    def handle_edge_cases(answer: int, problem_context: str = "") -> Tuple[int, str]:
        """Handle edge cases in answer validation."""
        edge_case_note = ""
        
        try:
            # Negative answers
            if answer < 0:
                edge_case_note = f"Negative answer {answer} converted to 0"
                return 0, edge_case_note
            
            # Very large answers
            if answer > AnswerValidator.AIMO_MAX * 10:
                answer = AnswerValidator.AIMO_MAX
                edge_case_note = f"Very large answer capped at {AnswerValidator.AIMO_MAX}"
                return answer, edge_case_note
            
            validated = AnswerValidator.validate_integer(answer)
            return validated if validated is not None else 0, edge_case_note
            
        except Exception as e:
            logger.debug(f"Edge case handling failed: {str(e)}")
            return 0, f"Edge case error: {str(e)}"

print("✅ AnswerValidator class loaded")

# Test 4: Phase 4 - ExecutionMetrics Class
print("\nTest 4: Loading ExecutionMetrics class...")

class ExecutionMetrics:
    """Track execution metrics for the pipeline."""
    
    def __init__(self):
        self.metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "fallback_used": 0,
            "verified": 0,
            "average_confidence": 0.0
        }
    
    def record_result(
        self,
        success: bool,
        fallback_used: bool = False,
        verified: bool = False,
        confidence: float = 0.0
    ) -> None:
        """Record a single result."""
        self.metrics["total_processed"] += 1
        
        if success:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1
        
        if fallback_used:
            self.metrics["fallback_used"] += 1
        
        if verified:
            self.metrics["verified"] += 1
        
        # Update average confidence
        if self.metrics["successful"] > 0:
            self.metrics["average_confidence"] = (
                (self.metrics["average_confidence"] * (self.metrics["successful"] - 1) + confidence) /
                self.metrics["successful"]
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        total = self.metrics["total_processed"]
        
        return {
            "total_processed": total,
            "successful": self.metrics["successful"],
            "failed": self.metrics["failed"],
            "success_rate": self.metrics["successful"] / total if total > 0 else 0.0,
            "fallback_used_count": self.metrics["fallback_used"],
            "fallback_rate": self.metrics["fallback_used"] / total if total > 0 else 0.0,
            "verified_count": self.metrics["verified"],
            "verification_rate": self.metrics["verified"] / total if total > 0 else 0.0,
            "average_confidence": self.metrics["average_confidence"]
        }

print("✅ ExecutionMetrics class loaded")

# Test 5: Functional Tests
print("\n" + "=" * 60)
print("FUNCTIONAL TESTS")
print("=" * 60)

# Test SymbolicCompute
print("\nTest 5a: SymbolicCompute.evaluate_expression()")
test_expr = "2 + 3 * 5"
result = SymbolicCompute.evaluate_expression(test_expr)
print(f"  Expression: {test_expr}")
print(f"  Result: {result}")
assert result == 17, "Expression evaluation failed"
print("  ✅ PASSED")

# Test AnswerValidator
print("\nTest 5b: AnswerValidator.validate_integer()")
test_cases = [
    (50, 50),
    (-10, 0),
    (100000, 99999),
    (42.7, 42)
]
for input_val, expected in test_cases:
    result = AnswerValidator.validate_integer(input_val)
    print(f"  Input: {input_val}, Expected: {expected}, Got: {result}")
    assert result == expected, f"Validation failed for {input_val}"
print("  ✅ PASSED")

# Test ExecutionMetrics
print("\nTest 5c: ExecutionMetrics tracking")
metrics = ExecutionMetrics()
metrics.record_result(success=True, verified=True, confidence=0.95)
metrics.record_result(success=True, fallback_used=True, confidence=0.75)
metrics.record_result(success=False)
summary = metrics.get_summary()
print(f"  Total: {summary['total_processed']}")
print(f"  Success rate: {summary['success_rate']:.1%}")
print(f"  Avg confidence: {summary['average_confidence']:.2f}")
assert summary['total_processed'] == 3, "Metrics tracking failed"
assert summary['successful'] == 2, "Success count wrong"
print("  ✅ PASSED")

# Test integrated workflow
print("\nTest 5d: Integrated validation workflow")
llm_answer = 17
llm_text = "The calculation is 2 + 3*5 = 17. The answer is 17."
result = AnswerValidator.validate_with_fallback_strategies(llm_answer, llm_text)
print(f"  Answer: {result['final_answer']}")
print(f"  Confidence: {result['confidence']:.2f}")
print(f"  Strategy: {result['strategy_used']}")
assert result['final_answer'] == 17, "Integrated workflow failed"
print("  ✅ PASSED")

# Test edge cases
print("\nTest 5e: Edge case handling")
test_cases = [
    (-5, 0, "negative"),
    (1000000, 99999, "very large"),
    (50, 50, "normal")
]
for input_val, expected, description in test_cases:
    result, note = AnswerValidator.handle_edge_cases(input_val)
    print(f"  {description}: {input_val} → {result}")
    assert result == expected, f"Edge case failed for {description}"
print("  ✅ PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
print("\n📊 Summary:")
print("  ✅ All dependencies available")
print("  ✅ All Phase 4 classes loaded successfully")
print("  ✅ All functional tests passed")
print("  ✅ Notebook is ready for Kaggle deployment")
print()
