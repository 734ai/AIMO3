"""
pipeline.py - Complete AIMO3 Solver Pipeline

Orchestrates the entire workflow: preprocessing → reasoning → computation → postprocessing.
Designed for both batch processing and single problem solving.

Phase 4 Integration:
- Symbolic verification of LLM answers
- Fallback validation strategies
- Edge case handling
- Verification tracking and metrics
"""

import logging
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .preprocessing import prepare_problem, batch_prepare_problems
from .reasoning import LLMSolver, extract_numeric_answer
from .computation import SymbolicCompute, AnswerValidator
from .postprocessing import (
    SubmissionFormatter, ReasoningLogger, ResultsAggregator,
    VerificationTracker, ErrorRecoveryHandler, ExecutionMetrics
)


logger = logging.getLogger(__name__)


class AIMO3Pipeline:
    """
    Complete pipeline for solving AIMO3 problems.
    """
    
    def __init__(
        self,
        model_name: str = "Open-Orca/orca_mini_3b",
        device: str = "auto",
        output_dir: str = "outputs",
        log_dir: str = "logs"
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_name: HuggingFace model identifier for LLM
            device: Device to use for model
            output_dir: Directory for output files
            log_dir: Directory for log files
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        logger.info("Initializing AIMO3 Pipeline...")
        
        try:
            self.solver = LLMSolver(model_name=model_name, device=device)
            self.formatter = SubmissionFormatter(output_dir=output_dir)
            self.logger_inst = ReasoningLogger(log_dir=log_dir)
            self.compute = SymbolicCompute()
            self.validator = AnswerValidator()
            
            # Phase 4: Initialize verification components
            self.verification_tracker = VerificationTracker(output_dir=output_dir)
            self.error_recovery = ErrorRecoveryHandler()
            self.metrics = ExecutionMetrics()
            
            logger.info("Pipeline initialized successfully (Phase 4 components enabled)")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    
    def solve_single_problem(
        self,
        problem_text: str,
        problem_id: str = "P0",
        input_type: str = "text",
        max_tokens: int = 512,
        verify: bool = True,
        use_phase4_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Solve a single problem end-to-end with Phase 4 verification.
        
        Phase 4 Integration: Includes symbolic verification, fallback strategies,
        edge case handling, and verification tracking.
        
        Args:
            problem_text: The problem to solve
            problem_id: Identifier for the problem
            input_type: Type of input ("text", "latex", "pdf")
            max_tokens: Maximum tokens for LLM generation
            verify: Whether to verify answer with SymPy
            use_phase4_verification: Enable Phase 4 verification workflow
            
        Returns:
            Dictionary with problem, reasoning, answer, and metadata
        """
        result = {
            "problem_id": problem_id,
            "input_type": input_type,
            "status": "processing",
            "verification_workflow": "phase4" if use_phase4_verification else "basic"
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Preprocess
            logger.info(f"[{problem_id}] Preprocessing problem...")
            prepared_problem = prepare_problem(problem_text, input_type=input_type)
            result["prepared_problem"] = prepared_problem
            
            # Step 2: Reasoning with LLM
            logger.info(f"[{problem_id}] Running LLM reasoning...")
            reasoning_result = self.solver.solve(
                prepared_problem,
                max_tokens=max_tokens
            )
            result["raw_reasoning"] = reasoning_result["reasoning"]
            llm_text = reasoning_result["reasoning"]
            
            # Step 3: Extract numeric answer
            logger.info(f"[{problem_id}] Extracting numeric answer...")
            answer = extract_numeric_answer(reasoning_result["reasoning"])
            result["llm_extracted_answer"] = answer
            
            # Step 4: Phase 4 Enhanced Validation
            if use_phase4_verification and verify:
                logger.info(f"[{problem_id}] Phase 4: Enhanced verification workflow...")
                
                # Verify symbolically
                verification_result = self.compute.verify_symbolic_result(
                    answer if answer is not None else 0,
                    llm_text
                )
                result["symbolic_verification"] = verification_result
                
                # Validate with fallback strategies
                validation_result = self.validator.validate_with_fallback_strategies(
                    answer,
                    llm_text
                )
                result["validation_result"] = validation_result
                
                # Handle edge cases
                final_answer, edge_note = self.validator.handle_edge_cases(
                    validation_result.get("final_answer", 0),
                    prepared_problem
                )
                result["edge_case_handling"] = edge_note
                result["final_answer"] = final_answer
                
                # Track verification
                fallback_used = validation_result.get("fallback_applied", False)
                
                self.verification_tracker.log_verification(
                    problem_id,
                    answer,
                    final_answer,
                    verification_result,
                    fallback_used
                )
                
                # Record metrics
                self.metrics.record_result(
                    success=True,
                    fallback_used=fallback_used,
                    verified=verification_result.get("verification_passed", False),
                    confidence=validation_result.get("confidence", 0.0),
                    processing_time=time.time() - start_time
                )
            else:
                # Basic validation (Phase 1-3)
                logger.info(f"[{problem_id}] Basic validation...")
                validated_answer = self.validator.validate_integer(answer)
                result["final_answer"] = validated_answer
            
            result["status"] = "success"
            result["processing_time"] = time.time() - start_time
            
            logger.info(f"[{problem_id}] Problem solved. Answer: {result.get('final_answer')}")
            
        except Exception as e:
            logger.error(f"[{problem_id}] Error solving problem: {str(e)}")
            result["status"] = "error"
            result["error"] = str(e)
            result["processing_time"] = time.time() - start_time
            self.metrics.record_result(
                success=False,
                fallback_used=True,
                verified=False,
                confidence=0.0,
                processing_time=result["processing_time"]
            )
        
        return result
    
    
    def solve_batch(
        self,
        problems: List[str],
        problem_ids: List[str] = None,
        input_type: str = "text",
        max_tokens: int = 512,
        save_results: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Solve multiple problems in batch.
        
        Args:
            problems: List of problem texts
            problem_ids: List of problem IDs (auto-generated if None)
            input_type: Type of input for all problems
            max_tokens: Maximum tokens per problem
            save_results: Whether to save results to files
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with results summary and per-problem details
        """
        if problem_ids is None:
            problem_ids = [f"P{i}" for i in range(len(problems))]
        
        if len(problem_ids) != len(problems):
            raise ValueError("problem_ids and problems must have same length")
        
        logger.info(f"Starting batch processing of {len(problems)} problems")
        
        results = []
        predictions = []
        reasoning_steps = []
        
        iterator = tqdm(
            zip(problem_ids, problems),
            total=len(problems),
            disable=not verbose
        )
        
        for problem_id, problem_text in iterator:
            result = self.solve_single_problem(
                problem_text,
                problem_id=problem_id,
                input_type=input_type,
                max_tokens=max_tokens
            )
            
            results.append(result)
            predictions.append(result.get("final_answer", 0))
            reasoning_steps.append(result.get("raw_reasoning", ""))
            
            iterator.set_description(
                f"[{result['status'].upper()}] {problem_id}: "
                f"Answer={result.get('final_answer', '?')}"
            )
        
        # Compute statistics
        stats = ResultsAggregator.compute_statistics(
            problem_ids,
            predictions
        )
        
        batch_result = {
            "total_problems": len(problems),
            "statistics": stats,
            "results": results,
            "predictions": predictions,
            "reasoning_steps": reasoning_steps
        }
        
        # Save results if requested
        if save_results:
            logger.info("Saving results...")
            try:
                self.formatter.save_submission_csv(problem_ids, predictions)
                self.logger_inst.save_reasoning_steps(
                    problem_ids,
                    reasoning_steps,
                    predictions
                )
                self.logger_inst.save_statistics(stats)
                self.logger_inst.save_detailed_results(results)
            except Exception as e:
                logger.warning(f"Failed to save some results: {str(e)}")
        
        logger.info(f"Batch processing completed. Success rate: "
                   f"{stats.get('predictions_made', 0)}/{len(problems)}")
        
        return batch_result
    
    
    def solve_from_csv(
        self,
        csv_path: str,
        problem_column: str = "latex_problem",
        id_column: str = "problem_id",
        input_type: str = "latex",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Solve problems from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            problem_column: Name of column containing problems
            id_column: Name of column containing problem IDs
            input_type: Type of input in CSV
            **kwargs: Additional arguments passed to solve_batch
            
        Returns:
            Batch results dictionary
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except ImportError:
            raise ImportError("pandas is required for CSV input. Install with: pip install pandas")
        except Exception as e:
            logger.error(f"Failed to read CSV: {str(e)}")
            raise
        
        problems = df[problem_column].tolist()
        problem_ids = df[id_column].tolist() if id_column in df.columns else None
        
        logger.info(f"Loaded {len(problems)} problems from {csv_path}")
        
        return self.solve_batch(
            problems,
            problem_ids=problem_ids,
            input_type=input_type,
            **kwargs
        )


def main():
    """
    Example usage of the pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize pipeline
    pipeline = AIMO3Pipeline(
        model_name="Open-Orca/orca_mini_3b",
        device="auto"
    )
    
    # Example: Solve a single problem
    problem = "Compute 2 + 3 × 5. What is the final integer answer?"
    result = pipeline.solve_single_problem(problem, problem_id="Example1")
    
    print("\n" + "="*60)
    print("SINGLE PROBLEM RESULT")
    print("="*60)
    print(f"Problem: {problem}")
    print(f"Final Answer: {result.get('final_answer', 'Error')}")
    print(f"Processing Time: {result.get('processing_time', 'N/A'):.2f}s")
    
    # Example: Solve batch
    print("\n" + "="*60)
    print("BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    batch_problems = [
        "Solve: 2x + 5 = 13. What is x?",
        "Compute: 10 + 20 - 5",
        "Find: 7 × 8"
    ]
    
    batch_result = pipeline.solve_batch(
        batch_problems,
        problem_ids=["Q1", "Q2", "Q3"],
        save_results=True
    )
    
    print(f"\nBatch Results Summary:")
    print(f"Total Problems: {batch_result['total_problems']}")
    print(f"Successful: {batch_result['statistics']['predictions_made']}")
    print(f"Failed: {batch_result['statistics']['failed_predictions']}")
    print(f"Predictions: {batch_result['predictions']}")


if __name__ == "__main__":
    main()
