"""
computation.py - Symbolic and Numeric Computation Module

Integrates SymPy for symbolic computation and numeric verification
of mathematical expressions and answers.

Phase 4 Features:
- LLM output parsing for mathematical expressions
- Symbolic result verification
- Modular arithmetic support with large numbers
- Answer validation with edge case handling
"""

import re
import logging
from typing import Optional, Union, Any, Tuple, Dict

try:
    import sympy as sp
except ImportError:
    raise ImportError("sympy is required. Install with: pip install sympy")


logger = logging.getLogger(__name__)


class SymbolicCompute:
    """
    Wrapper for SymPy-based symbolic computation.
    """
    
    @staticmethod
    def evaluate_expression(expr_str: str) -> Optional[Union[int, float]]:
        """
        Evaluate a mathematical expression string using SymPy.
        
        Args:
            expr_str: Mathematical expression string
            
        Returns:
            Numeric result or None if evaluation fails
        """
        try:
            expr = sp.sympify(expr_str)
            result = expr.evalf()
            return int(result) if result == int(result) else float(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate expression '{expr_str}': {str(e)}")
            return None
    
    
    @staticmethod
    def simplify_expression(expr_str: str) -> Optional[str]:
        """
        Simplify a mathematical expression.
        
        Args:
            expr_str: Mathematical expression string
            
        Returns:
            Simplified expression string or None
        """
        try:
            expr = sp.sympify(expr_str)
            simplified = sp.simplify(expr)
            return str(simplified)
        except Exception as e:
            logger.warning(f"Failed to simplify expression '{expr_str}': {str(e)}")
            return None
    
    
    @staticmethod
    def solve_equation(equation_str: str, variable_str: str = "x") -> Optional[list]:
        """
        Solve an equation for a variable.
        
        Args:
            equation_str: Equation string (e.g., "2*x + 5 = 13")
            variable_str: Variable to solve for
            
        Returns:
            List of solutions or None
        """
        try:
            var = sp.Symbol(variable_str)
            
            # Parse equation - handle both "a = b" and "a - b" formats
            if "=" in equation_str:
                left, right = equation_str.split("=")
                equation = sp.sympify(left) - sp.sympify(right)
            else:
                equation = sp.sympify(equation_str)
            
            solutions = sp.solve(equation, var)
            return [float(sol) if isinstance(sol, (int, float, sp.Number)) else sol 
                    for sol in solutions]
        except Exception as e:
            logger.warning(f"Failed to solve equation '{equation_str}': {str(e)}")
            return None
    
    
    @staticmethod
    def compute_modular_arithmetic(value: int, modulus: int) -> int:
        """
        Compute modular arithmetic result.
        
        Args:
            value: Value to compute modulo
            modulus: Modulus
            
        Returns:
            Result of value mod modulus
        """
        try:
            return int(value) % int(modulus)
        except Exception as e:
            logger.warning(f"Failed modular computation: {str(e)}")
            return None
    
    
    @staticmethod
    def verify_answer(expr_str: str, expected_answer: int, tolerance: float = 1e-6) -> bool:
        """
        Verify if computed result matches expected answer.
        
        Args:
            expr_str: Expression to evaluate
            expected_answer: Expected numeric answer
            tolerance: Tolerance for floating-point comparison
            
        Returns:
            True if result matches expected answer
        """
        try:
            result = SymbolicCompute.evaluate_expression(expr_str)
            if result is None:
                return False
            
            if isinstance(result, float):
                return abs(result - expected_answer) < tolerance
            else:
                return int(result) == expected_answer
        except Exception as e:
            logger.warning(f"Failed verification: {str(e)}")
            return False
    
    
    @staticmethod
    def parse_llm_output_for_expressions(llm_text: str) -> Dict[str, Any]:
        """
        Parse LLM output to extract mathematical expressions and computations.
        
        Phase 4: Enhanced LLM output parsing for verification
        
        Args:
            llm_text: Raw text output from LLM
            
        Returns:
            Dictionary with extracted expressions, values, and equations
        """
        result = {
            "raw_text": llm_text,
            "expressions": [],
            "values": [],
            "equations": [],
            "final_value": None
        }
        
        try:
            # Extract mathematical expressions (patterns like "x + y", "2*3 + 5", etc.)
            # Match expressions with operators
            expr_pattern = r"(?:=|equals|is)\s*(\d+(?:\.\d+)?|[\w\s\+\-\*/\(\)\.]+)"
            expr_matches = re.findall(expr_pattern, llm_text, re.IGNORECASE)
            
            # Extract equations
            eq_pattern = r"([a-zA-Z]\s*=\s*[\d\+\-\*/\(\)\.]+)"
            eq_matches = re.findall(eq_pattern, llm_text)
            
            # Extract numeric values
            num_pattern = r"\b(\d+(?:\.\d+)?)\b"
            num_matches = re.findall(num_pattern, llm_text)
            
            result["expressions"] = expr_matches[:5]  # Keep first 5
            result["equations"] = eq_matches[:5]       # Keep first 5
            result["values"] = [float(n) if '.' in n else int(n) for n in num_matches[:10]]
            
            # Try to extract final numeric value
            if num_matches:
                result["final_value"] = float(num_matches[-1]) if '.' in num_matches[-1] else int(num_matches[-1])
                
        except Exception as e:
            logger.warning(f"Failed to parse LLM output: {str(e)}")
        
        return result
    
    
    @staticmethod
    def verify_symbolic_result(
        llm_answer: int,
        llm_output: str,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Verify LLM answer by parsing expressions from output and computing symbolically.
        
        Phase 4: LLM output verification workflow
        
        Args:
            llm_answer: The answer extracted from LLM output
            llm_output: Full LLM output text for parsing
            tolerance: Tolerance for comparison (as percentage, 0-1)
            
        Returns:
            Dictionary with verification results and confidence
        """
        verification = {
            "llm_answer": llm_answer,
            "symbolic_result": None,
            "expressions_found": [],
            "verification_passed": False,
            "confidence": 0.0,
            "details": ""
        }
        
        try:
            # Parse expressions from LLM output
            parsed = SymbolicCompute.parse_llm_output_for_expressions(llm_output)
            verification["expressions_found"] = parsed["expressions"]
            
            # Try to evaluate extracted expressions
            if parsed["expressions"]:
                for expr_str in parsed["expressions"]:
                    try:
                        result = SymbolicCompute.evaluate_expression(expr_str)
                        if result is not None:
                            # Compare with LLM answer
                            if isinstance(result, float):
                                diff_percent = abs(result - llm_answer) / max(abs(llm_answer), 1)
                                if diff_percent <= tolerance:
                                    verification["symbolic_result"] = result
                                    verification["verification_passed"] = True
                                    verification["confidence"] = 1.0 - diff_percent
                                    verification["details"] = f"Symbolic computation matches LLM answer within {diff_percent*100:.1f}%"
                                    break
                            else:
                                if int(result) == llm_answer:
                                    verification["symbolic_result"] = result
                                    verification["verification_passed"] = True
                                    verification["confidence"] = 1.0
                                    verification["details"] = "Symbolic computation exactly matches LLM answer"
                                    break
                    except Exception as e:
                        continue
            
            # If no exact match found, increase confidence in final value
            if not verification["verification_passed"] and parsed["final_value"] is not None:
                if parsed["final_value"] == llm_answer:
                    verification["confidence"] = 0.8
                    verification["details"] = "Final extracted value matches LLM answer"
                    
        except Exception as e:
            logger.warning(f"Verification failed: {str(e)}")
            verification["details"] = f"Verification error: {str(e)}"
        
        return verification
    
    
    @staticmethod
    def compute_with_large_numbers(expr_str: str, max_result: int = 99999) -> Optional[int]:
        """
        Compute expressions with large numbers, handling potential overflow.
        
        Phase 4: Large number computation with modular arithmetic awareness
        
        Args:
            expr_str: Expression string
            max_result: Maximum expected result (AIMO max is 99999)
            
        Returns:
            Computed result or None if evaluation fails
        """
        try:
            expr = sp.sympify(expr_str)
            result = expr.evalf()
            
            # Convert to integer
            if result == int(result):
                result_int = int(result)
            else:
                result_int = int(round(float(result)))
            
            # Handle potential overflow - if result exceeds max, check for modular arithmetic
            if result_int > max_result:
                logger.warning(f"Result {result_int} exceeds max {max_result}, checking for mod patterns")
                # Could be a modular arithmetic problem
                result_int = result_int % (max_result + 1)  # Mod to 0-99999 range
            
            return result_int
            
        except Exception as e:
            logger.warning(f"Failed large number computation for '{expr_str}': {str(e)}")
            return None


class AnswerValidator:
    """
    Validates and enforces answer format constraints.
    """
    
    AIMO_MIN = 0
    AIMO_MAX = 99999
    
    @staticmethod
    def validate_integer(value: Any) -> Optional[int]:
        """
        Validate and convert value to valid AIMO integer.
        
        Args:
            value: Value to validate
            
        Returns:
            Integer in valid AIMO range or None
        """
        try:
            int_value = int(float(str(value).strip()))
            
            if int_value < AnswerValidator.AIMO_MIN:
                logger.warning(f"Answer {int_value} below minimum {AnswerValidator.AIMO_MIN}")
                return AnswerValidator.AIMO_MIN
            
            if int_value > AnswerValidator.AIMO_MAX:
                logger.warning(f"Answer {int_value} above maximum {AnswerValidator.AIMO_MAX}")
                return AnswerValidator.AIMO_MAX
            
            return int_value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to validate answer: {str(e)}")
            return None
    
    
    @staticmethod
    def extract_and_validate_answer(text: str) -> Optional[int]:
        """
        Extract numeric answer from text and validate it.
        
        Args:
            text: Text to extract answer from
            
        Returns:
            Validated integer answer or None
        """
        # Look for patterns indicating final answer
        patterns = [
            r"(?:answer|result|final answer)\s*:?\s*(\d+)",
            r"(?:the answer is|equals)\s*(\d+)",
            r"(\d{1,5})\s*(?:is the answer|is correct)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    candidate = int(match.group(1))
                    validated = AnswerValidator.validate_integer(candidate)
                    if validated is not None:
                        return validated
                except ValueError:
                    continue
        
        # Fallback: find all numbers and validate the last one
        numbers = re.findall(r"\d+", text)
        if numbers:
            return AnswerValidator.validate_integer(numbers[-1])
        
        return None
    
    
    @staticmethod
    def validate_with_fallback_strategies(
        llm_answer: Optional[int],
        llm_text: str,
        fallback_order: str = "symbolic_first"
    ) -> Dict[str, Any]:
        """
        Validate answer with multiple fallback strategies.
        
        Phase 4: Enhanced validation with edge case handling
        
        Args:
            llm_answer: Initial LLM-extracted answer
            llm_text: Full LLM output for secondary extraction
            fallback_order: Strategy order ("symbolic_first" or "extraction_first")
            
        Returns:
            Dictionary with validated answer and confidence
        """
        result = {
            "final_answer": None,
            "confidence": 0.0,
            "strategy_used": "none",
            "fallback_applied": False,
            "details": ""
        }
        
        try:
            # Strategy 1: Use primary answer if valid
            if llm_answer is not None:
                validated = AnswerValidator.validate_integer(llm_answer)
                if validated is not None:
                    result["final_answer"] = validated
                    result["confidence"] = 0.9
                    result["strategy_used"] = "primary_llm_answer"
                    result["details"] = f"Primary LLM answer {llm_answer} validated"
                    return result
            
            # Strategy 2: Try symbolic verification
            sym_result = SymbolicCompute.verify_symbolic_result(
                llm_answer if llm_answer is not None else 0,
                llm_text
            )
            
            if sym_result["verification_passed"] and sym_result["symbolic_result"] is not None:
                validated = AnswerValidator.validate_integer(int(sym_result["symbolic_result"]))
                if validated is not None:
                    result["final_answer"] = validated
                    result["confidence"] = sym_result["confidence"]
                    result["strategy_used"] = "symbolic_verification"
                    result["fallback_applied"] = True
                    result["details"] = sym_result["details"]
                    return result
            
            # Strategy 3: Re-extract from text
            extracted = AnswerValidator.extract_and_validate_answer(llm_text)
            if extracted is not None:
                result["final_answer"] = extracted
                result["confidence"] = 0.75
                result["strategy_used"] = "text_reextraction"
                result["fallback_applied"] = True
                result["details"] = f"Answer re-extracted from text: {extracted}"
                return result
            
            # Strategy 4: Default fallback
            result["final_answer"] = 0
            result["confidence"] = 0.0
            result["strategy_used"] = "default_fallback"
            result["fallback_applied"] = True
            result["details"] = "No valid answer found, using default: 0"
            
        except Exception as e:
            logger.warning(f"Fallback validation failed: {str(e)}")
            result["final_answer"] = 0
            result["details"] = f"Error in fallback validation: {str(e)}"
        
        return result
    
    
    @staticmethod
    def handle_edge_cases(
        answer: int,
        problem_context: str = ""
    ) -> Tuple[int, str]:
        """
        Handle edge cases in answer validation.
        
        Phase 4: Edge case handling for robustness
        
        Args:
            answer: The answer to validate
            problem_context: Context string for detecting problem types
            
        Returns:
            Tuple of (validated_answer, edge_case_note)
        """
        edge_case_note = ""
        
        try:
            # Edge case 1: Negative answers (not valid in AIMO range)
            if answer < 0:
                edge_case_note = f"Negative answer {answer} converted to 0"
                return 0, edge_case_note
            
            # Edge case 2: Very large answers (likely computation error)
            if answer > AnswerValidator.AIMO_MAX * 10:
                # Check if it's a modular arithmetic problem
                if "mod" in problem_context.lower():
                    modulus = 1000  # Common modulus
                    answer = answer % modulus
                    edge_case_note = f"Large answer modded to {answer}"
                    return answer, edge_case_note
                else:
                    answer = AnswerValidator.AIMO_MAX
                    edge_case_note = f"Very large answer capped at maximum {AnswerValidator.AIMO_MAX}"
                    return answer, edge_case_note
            
            # Edge case 3: Floating point results from division
            # (handled in validate_integer but documented here)
            
            # Edge case 4: Probabilistic answers (0-1 range)
            if 0 < answer <= 1 and "probability" in problem_context.lower():
                # Might need scaling, but keep as-is for now
                edge_case_note = "Probability answer kept in 0-1 range"
            
            validated = AnswerValidator.validate_integer(answer)
            return validated if validated is not None else 0, edge_case_note
            
        except Exception as e:
            logger.warning(f"Edge case handling failed: {str(e)}")
            return 0, f"Edge case error: {str(e)}"


def format_answer_for_submission(answer: int) -> str:
    """
    Format answer for Kaggle submission.
    
    Args:
        answer: Integer answer
        
    Returns:
        Formatted answer string
    """
    validated = AnswerValidator.validate_integer(answer)
    return str(validated) if validated is not None else "0"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test expression evaluation
    compute = SymbolicCompute()
    
    result = compute.evaluate_expression("2 + 3 * 5")
    print(f"2 + 3 * 5 = {result}")
    
    # Test equation solving
    solutions = compute.solve_equation("2*x + 5 = 13", "x")
    print(f"Solutions to 2*x + 5 = 13: {solutions}")
    
    # Test answer validation
    validator = AnswerValidator()
    valid_answer = validator.validate_integer(42)
    print(f"Valid answer: {valid_answer}")
    
    # Test extraction and validation
    text = "After working through the steps, the final answer is 17"
    extracted = validator.extract_and_validate_answer(text)
    print(f"Extracted answer from text: {extracted}")
