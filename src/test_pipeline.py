"""
test_pipeline.py - Comprehensive Test Suite for AIMO3 Pipeline

Tests all modules including preprocessing, reasoning, computation, and pipeline integration.
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import latex_to_text, prepare_problem
from computation import SymbolicCompute, AnswerValidator
from postprocessing import SubmissionFormatter, ReasoningLogger, ResultsAggregator
from data_preparation import (
    DatasetLoader, DataPreprocessor, SyntheticDataGenerator, LaTeXValidator
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestResults:
    """Container for test results."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add(self, name: str, passed: bool, message: str = ""):
        """Add test result."""
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for test in self.tests:
            status = "✅ PASS" if test["passed"] else "❌ FAIL"
            print(f"{status}: {test['name']}")
            if test["message"]:
                print(f"  └─ {test['message']}")
        
        total = self.passed + self.failed
        print("\n" + "-"*60)
        print(f"Total: {total} | Passed: {self.passed} | Failed: {self.failed}")
        
        if self.failed == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print(f"❌ {self.failed} TEST(S) FAILED")
        print("="*60 + "\n")
        
        return self.failed == 0


def test_preprocessing():
    """Test preprocessing module."""
    results = TestResults()
    logger.info("Testing preprocessing module...")
    
    # Test 1: LaTeX to text conversion
    try:
        latex = r"$2 + 3 \times 5$"
        text = latex_to_text(latex)
        passed = isinstance(text, str) and len(text) > 0
        results.add("LaTeX to text conversion", passed, f"Result: {text}")
    except Exception as e:
        results.add("LaTeX to text conversion", False, str(e))
    
    # Test 2: Complex LaTeX
    try:
        latex = r"\text{Solve } 2x + 5 = 13"
        text = latex_to_text(latex)
        passed = "Solve" in text and "2x" in text
        results.add("Complex LaTeX parsing", passed, f"Result: {text}")
    except Exception as e:
        results.add("Complex LaTeX parsing", False, str(e))
    
    # Test 3: Plain text passthrough
    try:
        text = "Compute 2 + 3 × 5"
        prepared = prepare_problem(text, input_type="text")
        passed = prepared == text
        results.add("Plain text passthrough", passed)
    except Exception as e:
        results.add("Plain text passthrough", False, str(e))
    
    return results


def test_computation():
    """Test computation module."""
    results = TestResults()
    logger.info("Testing computation module...")
    
    compute = SymbolicCompute()
    
    # Test 1: Simple arithmetic
    try:
        result = compute.evaluate_expression("2 + 3 * 5")
        passed = result == 17
        results.add("Simple arithmetic (2 + 3*5 = 17)", passed, f"Result: {result}")
    except Exception as e:
        results.add("Simple arithmetic", False, str(e))
    
    # Test 2: Power operation
    try:
        result = compute.evaluate_expression("2**10")
        passed = result == 1024
        results.add("Power operation (2^10 = 1024)", passed, f"Result: {result}")
    except Exception as e:
        results.add("Power operation", False, str(e))
    
    # Test 3: Answer validation
    try:
        validator = AnswerValidator()
        answer = validator.validate_integer(150000)
        passed = answer == 99999  # Should clamp to max
        results.add("Answer clamping (150000 → 99999)", passed, f"Result: {answer}")
    except Exception as e:
        results.add("Answer clamping", False, str(e))
    
    # Test 4: Negative answer validation
    try:
        validator = AnswerValidator()
        answer = validator.validate_integer(-50)
        passed = answer == 0  # Should clamp to min
        results.add("Answer clamping (-50 → 0)", passed, f"Result: {answer}")
    except Exception as e:
        results.add("Answer clamping negative", False, str(e))
    
    # Test 5: Valid range answer
    try:
        validator = AnswerValidator()
        answer = validator.validate_integer(42)
        passed = answer == 42  # Should stay same
        results.add("Valid answer passthrough (42 → 42)", passed, f"Result: {answer}")
    except Exception as e:
        results.add("Valid answer passthrough", False, str(e))
    
    return results


def test_synthetic_data():
    """Test synthetic data generation."""
    results = TestResults()
    logger.info("Testing synthetic data generation...")
    
    # Test 1: Arithmetic problems
    try:
        problems = SyntheticDataGenerator.generate_arithmetic_problems(count=5)
        passed = len(problems) == 5 and all("problem_id" in p for p in problems)
        results.add("Arithmetic problem generation", passed, f"Generated {len(problems)} problems")
    except Exception as e:
        results.add("Arithmetic problem generation", False, str(e))
    
    # Test 2: Algebra problems
    try:
        problems = SyntheticDataGenerator.generate_algebra_problems(count=5)
        passed = len(problems) > 0 and all("answer" in p for p in problems)
        results.add("Algebra problem generation", passed, f"Generated {len(problems)} problems")
    except Exception as e:
        results.add("Algebra problem generation", False, str(e))
    
    # Test 3: Combinatorics problems
    try:
        problems = SyntheticDataGenerator.generate_combinatorics_problems(count=5)
        passed = len(problems) == 5 and all(p["answer"] > 0 for p in problems)
        results.add("Combinatorics problem generation", passed, f"Generated {len(problems)} problems")
    except Exception as e:
        results.add("Combinatorics problem generation", False, str(e))
    
    # Test 4: All categories
    try:
        problems = SyntheticDataGenerator.generate_all_synthetic(count_per_category=3)
        passed = len(problems) == 9  # 3 categories * 3 problems
        results.add("Multi-category generation", passed, f"Generated {len(problems)} total problems")
    except Exception as e:
        results.add("Multi-category generation", False, str(e))
    
    return results


def test_data_preprocessing():
    """Test data preprocessing functions."""
    results = TestResults()
    logger.info("Testing data preprocessing...")
    
    import pandas as pd
    
    # Create test dataframe
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'problem': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'answer': [10, 20, 30, 40, 50]
    })
    
    # Test 1: Column validation
    try:
        valid = DataPreprocessor.validate_columns(test_df, ['id', 'problem'])
        results.add("Column validation (valid)", valid)
    except Exception as e:
        results.add("Column validation (valid)", False, str(e))
    
    # Test 2: Column validation (missing)
    try:
        valid = not DataPreprocessor.validate_columns(test_df, ['missing_col'])
        results.add("Column validation (missing detection)", valid)
    except Exception as e:
        results.add("Column validation (missing)", False, str(e))
    
    # Test 3: Data splitting
    try:
        train, val, test = DataPreprocessor.create_splits(test_df, 0.6, 0.2, 0.2)
        passed = len(train) == 3 and len(val) == 1 and len(test) == 1
        results.add("Data splitting (6/2/2)", passed, 
                   f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    except Exception as e:
        results.add("Data splitting", False, str(e))
    
    return results


def test_postprocessing():
    """Test postprocessing module."""
    results = TestResults()
    logger.info("Testing postprocessing module...")
    
    # Create temp directories
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Test 1: Submission formatting
    try:
        formatter = SubmissionFormatter(output_dir=temp_dir)
        problem_ids = ['P1', 'P2', 'P3']
        predictions = [42, 100, 17]
        
        output_path = formatter.save_submission_csv(
            problem_ids, predictions, output_file='test_submission.csv'
        )
        
        passed = os.path.exists(output_path)
        results.add("Submission CSV generation", passed, f"Saved to {output_path}")
    except Exception as e:
        results.add("Submission CSV generation", False, str(e))
    
    # Test 2: Statistics computation
    try:
        stats = ResultsAggregator.compute_statistics(
            ['P1', 'P2', 'P3'],
            [42, 100, 17]
        )
        passed = stats['total_problems'] == 3
        results.add("Statistics computation", passed, f"Stats: {stats}")
    except Exception as e:
        results.add("Statistics computation", False, str(e))
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    
    return results


def test_latex_validation():
    """Test LaTeX validation."""
    results = TestResults()
    logger.info("Testing LaTeX validation...")
    
    validator = LaTeXValidator()
    
    # Test 1: Valid LaTeX
    try:
        valid = validator.validate_latex(r"\frac{1}{2}")
        results.add("Valid LaTeX detection", valid)
    except Exception as e:
        results.add("Valid LaTeX detection", False, str(e))
    
    # Test 2: Unbalanced braces
    try:
        invalid = not validator.validate_latex(r"\frac{1}{2")
        results.add("Unbalanced brace detection", invalid)
    except Exception as e:
        results.add("Unbalanced brace detection", False, str(e))
    
    return results


def test_phase4_sympy_verification():
    """Test Phase 4: SymPy Verification features."""
    results = TestResults()
    logger.info("Testing Phase 4: SymPy Verification...")
    
    compute = SymbolicCompute()
    
    # Test 1: Parse LLM output for expressions
    try:
        llm_text = """
        Let me solve this step by step.
        First, I compute 2 + 3 = 5
        Then, 5 * 4 = 20
        So the answer is 20
        """
        parsed = compute.parse_llm_output_for_expressions(llm_text)
        passed = (
            isinstance(parsed, dict) and
            "expressions" in parsed and
            "values" in parsed and
            parsed["final_value"] is not None
        )
        results.add(
            "Parse LLM output for expressions",
            passed,
            f"Found {len(parsed.get('expressions', []))} expressions, final value: {parsed.get('final_value')}"
        )
    except Exception as e:
        results.add("Parse LLM output for expressions", False, str(e))
    
    # Test 2: Verify symbolic result
    try:
        llm_output = "2 + 3 * 5 equals 17"
        verification = compute.verify_symbolic_result(17, llm_output)
        passed = (
            isinstance(verification, dict) and
            "verification_passed" in verification and
            "confidence" in verification
        )
        results.add(
            "Verify symbolic result",
            passed,
            f"Confidence: {verification.get('confidence', 0):.2f}"
        )
    except Exception as e:
        results.add("Verify symbolic result", False, str(e))
    
    # Test 3: Large number computation with mod handling
    try:
        expr = "5000 * 50000"  # Large result
        result = compute.compute_with_large_numbers(expr)
        passed = result is not None and 0 <= result <= 99999
        results.add(
            "Large number computation with modular handling",
            passed,
            f"Result: {result} (within AIMO range)"
        )
    except Exception as e:
        results.add("Large number computation with modular handling", False, str(e))
    
    # Test 4: Large number exceeds max
    try:
        expr = "200000"  # Exceeds AIMO max
        result = compute.compute_with_large_numbers(expr)
        passed = result is not None and result <= 99999
        results.add(
            "Large number mod reduction",
            passed,
            f"200000 mod handled to: {result}"
        )
    except Exception as e:
        results.add("Large number mod reduction", False, str(e))
    
    return results


def test_phase4_answer_validation():
    """Test Phase 4: Enhanced Answer Validation."""
    results = TestResults()
    logger.info("Testing Phase 4: Answer Validation...")
    
    validator = AnswerValidator()
    
    # Test 1: Validate with fallback strategies
    try:
        llm_answer = None  # Simulate failed extraction
        llm_text = "After calculation, the final answer is 42"
        validation = validator.validate_with_fallback_strategies(llm_answer, llm_text)
        passed = (
            validation.get("final_answer") == 42 and
            validation.get("confidence") > 0 and
            validation.get("fallback_applied") == True
        )
        results.add(
            "Fallback validation with text extraction",
            passed,
            f"Answer: {validation.get('final_answer')}, Strategy: {validation.get('strategy_used')}"
        )
    except Exception as e:
        results.add("Fallback validation with text extraction", False, str(e))
    
    # Test 2: Edge case handling - negative answer
    try:
        answer, note = validator.handle_edge_cases(-50, "")
        passed = answer == 0 and "Negative" in note
        results.add(
            "Edge case: Negative answer",
            passed,
            f"Converted -50 to {answer}. Note: {note}"
        )
    except Exception as e:
        results.add("Edge case: Negative answer", False, str(e))
    
    # Test 3: Edge case handling - very large answer
    try:
        answer, note = validator.handle_edge_cases(1000000, "")
        passed = answer == 99999 and "capped" in note.lower()
        results.add(
            "Edge case: Very large answer",
            passed,
            f"Capped 1000000 to {answer}. Note: {note}"
        )
    except Exception as e:
        results.add("Edge case: Very large answer", False, str(e))
    
    # Test 4: Edge case handling - modular arithmetic problem
    try:
        answer, note = validator.handle_edge_cases(5000000, "Find 5000000 mod 1000")
        passed = answer < 99999 and ("mod" in note.lower() or "edg" in note.lower())
        results.add(
            "Edge case: Modular arithmetic",
            passed,
            f"Handled 5000000 to {answer}. Note: {note}"
        )
    except Exception as e:
        results.add("Edge case: Modular arithmetic", False, str(e))
    
    return results


def test_phase4_postprocessing():
    """Test Phase 4: Enhanced Postprocessing."""
    results = TestResults()
    logger.info("Testing Phase 4: Enhanced Postprocessing...")
    
    try:
        from postprocessing import (
            VerificationTracker, ErrorRecoveryHandler,
            ExecutionMetrics
        )
    except ImportError as e:
        results.add("Import Phase 4 postprocessing modules", False, str(e))
        return results
    
    # Test 1: Verification tracking
    try:
        tracker = VerificationTracker(output_dir="outputs")
        tracker.log_verification(
            "P1",
            llm_answer=42,
            final_answer=42,
            verification_result={"verified": True},
            fallback_applied=False
        )
        tracker.log_verification(
            "P2",
            llm_answer=50,
            final_answer=42,
            verification_result={"verified": False},
            fallback_applied=True
        )
        
        summary = tracker.get_summary_statistics()
        passed = (
            summary["total_problems"] == 2 and
            summary["fallback_applied_count"] == 1 and
            summary["answers_changed"] == 1
        )
        results.add(
            "Verification tracking",
            passed,
            f"Tracked 2 problems, 1 fallback, 1 changed"
        )
    except Exception as e:
        results.add("Verification tracking", False, str(e))
    
    # Test 2: Error recovery handling
    try:
        recovery = ErrorRecoveryHandler.recover_from_validation_failure(
            "P1",
            "Validation failed",
            fallback_value=0
        )
        passed = (
            recovery["recovery_attempted"] == True and
            recovery["recovered"] == True and
            recovery["fallback_answer"] == 0
        )
        results.add(
            "Error recovery handling",
            passed,
            f"Recovery strategy: {recovery['strategy']}"
        )
    except Exception as e:
        results.add("Error recovery handling", False, str(e))
    
    # Test 3: Timeout error handling
    try:
        timeout_result = ErrorRecoveryHandler.handle_timeout_error(
            "P1",
            current_best_answer=42
        )
        passed = (
            timeout_result["error_type"] == "timeout" and
            timeout_result["final_answer"] == 42
        )
        results.add(
            "Timeout error handling",
            passed,
            f"Timeout handled with best answer: {timeout_result['final_answer']}"
        )
    except Exception as e:
        results.add("Timeout error handling", False, str(e))
    
    # Test 4: Execution metrics tracking
    try:
        metrics = ExecutionMetrics()
        
        # Record some results
        metrics.record_result(success=True, fallback_used=False, verified=True, confidence=0.95, processing_time=0.5)
        metrics.record_result(success=True, fallback_used=True, verified=True, confidence=0.80, processing_time=0.6)
        metrics.record_result(success=False, fallback_used=True, verified=False, confidence=0.0, processing_time=0.4)
        
        summary = metrics.get_summary()
        passed = (
            summary["total_processed"] == 3 and
            summary["successful"] == 2 and
            summary["failed"] == 1 and
            summary["fallback_rate"] > 0.6
        )
        results.add(
            "Execution metrics tracking",
            passed,
            f"3 problems: {summary['successful']} success, {summary['failed']} failed, fallback_rate: {summary['fallback_rate']:.2f}"
        )
    except Exception as e:
        results.add("Execution metrics tracking", False, str(e))
    
    return results


def test_phase4_integration():
    """Test Phase 4: Full Integration."""
    results = TestResults()
    logger.info("Testing Phase 4: Full Integration...")
    
    compute = SymbolicCompute()
    validator = AnswerValidator()
    
    # Integration test: Complete verification workflow
    try:
        # Simulate LLM output
        llm_output = """
        Let me solve: Find the value of 2 * 3 + 7
        Step 1: 2 * 3 = 6
        Step 2: 6 + 7 = 13
        Therefore, the answer is 13
        """
        
        # Step 1: Verify symbolically
        verification = compute.verify_symbolic_result(13, llm_output)
        
        # Step 2: Validate with fallback
        validation = validator.validate_with_fallback_strategies(13, llm_output)
        
        # Step 3: Handle edge cases
        final_answer, edge_note = validator.handle_edge_cases(
            validation.get("final_answer", 0),
            llm_output
        )
        
        passed = (
            verification.get("verification_passed") == True and
            validation.get("final_answer") == 13 and
            final_answer == 13
        )
        
        results.add(
            "Full Phase 4 verification workflow",
            passed,
            f"Verified: {verification.get('verification_passed')}, "
            f"Final Answer: {final_answer}, "
            f"Confidence: {validation.get('confidence', 0):.2f}"
        )
    except Exception as e:
        results.add("Full Phase 4 verification workflow", False, str(e))
    
    return results


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("🧪 AIMO3 PIPELINE TEST SUITE")
    print("="*60 + "\n")
    
    all_results = []
    
    # Run test suites
    all_results.append(("Preprocessing", test_preprocessing()))
    all_results.append(("Computation", test_computation()))
    all_results.append(("Synthetic Data", test_synthetic_data()))
    all_results.append(("Data Preprocessing", test_data_preprocessing()))
    all_results.append(("Postprocessing", test_postprocessing()))
    all_results.append(("LaTeX Validation", test_latex_validation()))
    
    # Phase 4 tests
    all_results.append(("Phase 4: SymPy Verification", test_phase4_sympy_verification()))
    all_results.append(("Phase 4: Answer Validation", test_phase4_answer_validation()))
    all_results.append(("Phase 4: Postprocessing", test_phase4_postprocessing()))
    all_results.append(("Phase 4: Integration", test_phase4_integration()))
    
    # Print all results
    total_passed = 0
    total_failed = 0
    
    for suite_name, results in all_results:
        print(f"\n{'='*60}")
        print(f"SUITE: {suite_name}")
        print(f"{'='*60}")
        
        for test in results.tests:
            status = "✅" if test["passed"] else "❌"
            print(f"{status} {test['name']}")
            if test["message"]:
                print(f"   └─ {test['message']}")
        
        print(f"Result: {results.passed} passed, {results.failed} failed")
        total_passed += results.passed
        total_failed += results.failed
    
    # Final summary
    print("\n" + "="*60)
    print("📊 OVERALL TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n✅ ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ {total_failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
