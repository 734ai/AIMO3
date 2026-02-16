"""
postprocessing.py - Output Formatting and Submission Module

Handles answer formatting, validation, Kaggle submission generation,
and logging of reasoning steps.

Phase 4 Features:
- Enhanced validation workflow with fallback strategies
- Verification metadata tracking
- Error recovery and edge case handling
- Confidence scoring for predictions
"""

import os
import csv
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas not available. Some functions may not work.")


logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """
    Handles formatting and saving submission files for Kaggle.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize submission formatter.
        
        Args:
            output_dir: Directory to save submission files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    
    def save_submission_csv(
        self,
        problem_ids: List[str],
        predictions: List[int],
        output_file: str = "submission.csv"
    ) -> str:
        """
        Save predictions in Kaggle submission CSV format.
        
        Args:
            problem_ids: List of problem IDs
            predictions: List of predicted answers
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        if len(problem_ids) != len(predictions):
            raise ValueError("problem_ids and predictions must have same length")
        
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            if pd is not None:
                df = pd.DataFrame({
                    "problem_id": problem_ids,
                    "predicted_answer": predictions
                })
                df.to_csv(output_path, index=False)
            else:
                # Fallback to standard CSV writer
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["problem_id", "predicted_answer"])
                    for pid, pred in zip(problem_ids, predictions):
                        writer.writerow([pid, pred])
            
            logger.info(f"Submission saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save submission: {str(e)}")
            raise
    
    
    def save_predictions_json(
        self,
        predictions: Dict[str, Any],
        output_file: str = "predictions.json"
    ) -> str:
        """
        Save predictions as JSON for detailed analysis.
        
        Args:
            predictions: Dictionary of predictions with metadata
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=2)
            
            logger.info(f"Predictions saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save predictions JSON: {str(e)}")
            raise
    
    
    def save_submission_with_metadata(
        self,
        problem_ids: List[str],
        predictions: List[int],
        metadata: List[Dict[str, Any]],
        output_file: str = "submission_with_metadata.json"
    ) -> str:
        """
        Save submission with complete metadata for verification and debugging.
        
        Phase 4: Verification-aware submission format
        
        Args:
            problem_ids: List of problem IDs
            predictions: List of predicted answers
            metadata: List of metadata dictionaries (confidence, strategy, etc.)
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        if not (len(problem_ids) == len(predictions) == len(metadata)):
            raise ValueError("All lists must have the same length")
        
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            submission_data = []
            for pid, pred, meta in zip(problem_ids, predictions, metadata):
                submission_data.append({
                    "problem_id": pid,
                    "predicted_answer": pred,
                    "confidence": meta.get("confidence", 0.0),
                    "strategy_used": meta.get("strategy_used", "unknown"),
                    "verification_passed": meta.get("verification_passed", False),
                    "details": meta.get("details", "")
                })
            
            with open(output_path, "w") as f:
                json.dump(submission_data, f, indent=2)
            
            logger.info(f"Submission with metadata saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save submission with metadata: {str(e)}")
            raise


class VerificationTracker:
    """
    Tracks verification results and strategies used for each prediction.
    
    Phase 4: Verification workflow management
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize verification tracker.
        
        Args:
            output_dir: Directory to save verification logs
        """
        self.output_dir = output_dir
        self.verification_log = []
        os.makedirs(output_dir, exist_ok=True)
    
    
    def log_verification(
        self,
        problem_id: str,
        llm_answer: Optional[int],
        final_answer: int,
        verification_result: Dict[str, Any],
        fallback_applied: bool = False
    ) -> None:
        """
        Log verification result for a single problem.
        
        Args:
            problem_id: Problem identifier
            llm_answer: Initial LLM answer
            final_answer: Final validated answer
            verification_result: Verification result dictionary
            fallback_applied: Whether fallback strategy was used
        """
        log_entry = {
            "problem_id": problem_id,
            "timestamp": datetime.now().isoformat(),
            "llm_answer": llm_answer,
            "final_answer": final_answer,
            "verification_result": verification_result,
            "fallback_applied": fallback_applied,
            "answer_changed": llm_answer != final_answer if llm_answer is not None else False
        }
        
        self.verification_log.append(log_entry)
        logger.debug(f"Logged verification for {problem_id}")
    
    
    def save_verification_log(
        self,
        output_file: str = "verification_log.json"
    ) -> str:
        """
        Save complete verification log to file.
        
        Args:
            output_file: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # Compute summary statistics
            total = len(self.verification_log)
            fallback_count = sum(1 for log in self.verification_log if log["fallback_applied"])
            changed_count = sum(1 for log in self.verification_log if log["answer_changed"])
            
            summary = {
                "total_problems": total,
                "fallback_applied_count": fallback_count,
                "fallback_rate": fallback_count / total if total > 0 else 0,
                "answers_changed": changed_count,
                "change_rate": changed_count / total if total > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            log_data = {
                "summary": summary,
                "detailed_log": self.verification_log
            }
            
            with open(output_path, "w") as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Verification log saved to {output_path}")
            logger.info(f"Summary: {fallback_count}/{total} used fallback ({fallback_count/total*100:.1f}%)")
            
            return output_path
        except Exception as e:
            logger.error(f"Failed to save verification log: {str(e)}")
            raise
    
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of verification workflow.
        
        Returns:
            Dictionary with verification statistics
        """
        if not self.verification_log:
            return {
                "total_problems": 0,
                "fallback_applied_count": 0,
                "fallback_rate": 0.0,
                "answers_changed": 0,
                "change_rate": 0.0
            }
        
        total = len(self.verification_log)
        fallback_count = sum(1 for log in self.verification_log if log["fallback_applied"])
        changed_count = sum(1 for log in self.verification_log if log["answer_changed"])
        
        return {
            "total_problems": total,
            "fallback_applied_count": fallback_count,
            "fallback_rate": fallback_count / total if total > 0 else 0,
            "answers_changed": changed_count,
            "change_rate": changed_count / total if total > 0 else 0
        }


class ErrorRecoveryHandler:
    """
    Handles error recovery and fallback strategies for failed computations.
    
    Phase 4: Robust error handling
    """
    
    @staticmethod
    def recover_from_validation_failure(
        problem_id: str,
        error_message: str,
        fallback_value: int = 0
    ) -> Dict[str, Any]:
        """
        Handle validation failure with recovery strategy.
        
        Args:
            problem_id: Problem identifier
            error_message: Error message from validation
            fallback_value: Default value if recovery fails
            
        Returns:
            Recovery result dictionary
        """
        result = {
            "problem_id": problem_id,
            "recovery_attempted": True,
            "recovered": True,
            "fallback_answer": fallback_value,
            "error": error_message,
            "strategy": "default_fallback"
        }
        
        logger.warning(f"[{problem_id}] Validation failed: {error_message}. Using fallback value: {fallback_value}")
        
        return result
    
    
    @staticmethod
    def handle_timeout_error(
        problem_id: str,
        current_best_answer: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Handle timeout errors in computation.
        
        Args:
            problem_id: Problem identifier
            current_best_answer: Best answer found before timeout
            
        Returns:
            Timeout handling result
        """
        result = {
            "problem_id": problem_id,
            "error_type": "timeout",
            "recovery_strategy": "best_effort",
            "final_answer": current_best_answer if current_best_answer is not None else 0,
            "incomplete": current_best_answer is None
        }
        
        logger.warning(f"[{problem_id}] Timeout. Using best answer: {current_best_answer}")
        
        return result
    
    
    @staticmethod
    def handle_computation_error(
        problem_id: str,
        expression: str,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Handle symbolic computation errors gracefully.
        
        Args:
            problem_id: Problem identifier
            expression: Expression that failed
            error_message: Error details
            
        Returns:
            Error handling result
        """
        result = {
            "problem_id": problem_id,
            "error_type": "computation",
            "failed_expression": expression,
            "error_message": error_message,
            "fallback_strategy": "numeric_fallback",
            "recovery_attempted": True
        }
        
        logger.warning(f"[{problem_id}] Computation error for '{expression}': {error_message}")
        
        return result


class ExecutionMetrics:
    """
    Track and report execution metrics for the pipeline.
    
    Phase 4: Performance and reliability tracking
    """
    
    def __init__(self):
        """Initialize execution metrics tracker."""
        self.metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "fallback_used": 0,
            "verified": 0,
            "unverified": 0,
            "average_confidence": 0.0,
            "processing_times": []
        }
    
    
    def record_result(
        self,
        success: bool,
        fallback_used: bool = False,
        verified: bool = False,
        confidence: float = 0.0,
        processing_time: float = 0.0
    ) -> None:
        """
        Record a single result.
        
        Args:
            success: Whether computation was successful
            fallback_used: Whether fallback strategy was used
            verified: Whether answer was verified
            confidence: Confidence score (0-1)
            processing_time: Time taken in seconds
        """
        self.metrics["total_processed"] += 1
        
        if success:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1
        
        if fallback_used:
            self.metrics["fallback_used"] += 1
        
        if verified:
            self.metrics["verified"] += 1
        else:
            self.metrics["unverified"] += 1
        
        self.metrics["processing_times"].append(processing_time)
        
        # Update average confidence
        if self.metrics["successful"] > 0:
            self.metrics["average_confidence"] = (
                (self.metrics["average_confidence"] * (self.metrics["successful"] - 1) + confidence) /
                self.metrics["successful"]
            )
    
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.
        
        Returns:
            Summary dictionary
        """
        total = self.metrics["total_processed"]
        
        if not self.metrics["processing_times"]:
            avg_time = 0.0
            total_time = 0.0
        else:
            avg_time = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            total_time = sum(self.metrics["processing_times"])
        
        summary = {
            "total_processed": total,
            "successful": self.metrics["successful"],
            "failed": self.metrics["failed"],
            "success_rate": self.metrics["successful"] / total if total > 0 else 0.0,
            "fallback_used_count": self.metrics["fallback_used"],
            "fallback_rate": self.metrics["fallback_used"] / total if total > 0 else 0.0,
            "verified_count": self.metrics["verified"],
            "verification_rate": self.metrics["verified"] / total if total > 0 else 0.0,
            "average_confidence": self.metrics["average_confidence"],
            "average_processing_time": avg_time,
            "total_processing_time": total_time
        }
        
        return summary
    
    
    def save_metrics(self, output_path: str = "outputs/metrics.json") -> str:
        """
        Save metrics to file.
        
        Args:
            output_path: Path to save metrics
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            metrics_data = {
                "summary": self.get_summary(),
                "detailed_metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Metrics saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save metrics: {str(e)}")
            raise


class ReasoningLogger:
    """
    Logs and saves detailed reasoning steps for debugging and analysis.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize reasoning logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    def save_reasoning_steps(
        self,
        problem_ids: List[str],
        reasoning_list: List[str],
        answers: List[int],
        output_file: str = None
    ) -> str:
        """
        Save detailed reasoning steps for each problem.
        
        Args:
            problem_ids: List of problem IDs
            reasoning_list: List of reasoning texts
            answers: List of final answers
            output_file: Output filename (auto-generated if None)
            
        Returns:
            Path to saved log file
        """
        if output_file is None:
            output_file = f"reasoning_steps_{self.timestamp}.log"
        
        output_path = os.path.join(self.log_dir, output_file)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for pid, reasoning, answer in zip(problem_ids, reasoning_list, answers):
                    f.write(f"{'='*60}\n")
                    f.write(f"Problem ID: {pid}\n")
                    f.write(f"Final Answer: {answer}\n")
                    f.write(f"{'='*60}\n")
                    f.write(reasoning + "\n\n")
            
            logger.info(f"Reasoning steps saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save reasoning steps: {str(e)}")
            raise
    
    
    def save_detailed_results(
        self,
        results: List[Dict[str, Any]],
        output_file: str = None
    ) -> str:
        """
        Save detailed results including problem, reasoning, and answer.
        
        Args:
            results: List of result dictionaries
            output_file: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = f"detailed_results_{self.timestamp}.json"
        
        output_path = os.path.join(self.log_dir, output_file)
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed results saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save detailed results: {str(e)}")
            raise
    
    
    def save_statistics(
        self,
        stats: Dict[str, Any],
        output_file: str = None
    ) -> str:
        """
        Save execution statistics.
        
        Args:
            stats: Dictionary of statistics
            output_file: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            output_file = f"statistics_{self.timestamp}.json"
        
        output_path = os.path.join(self.log_dir, output_file)
        
        try:
            stats["timestamp"] = self.timestamp
            stats["saved_at"] = datetime.now().isoformat()
            
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Statistics saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save statistics: {str(e)}")
            raise


class ResultsAggregator:
    """
    Aggregates and analyzes results from the pipeline.
    """
    
    @staticmethod
    def compute_statistics(
        problem_ids: List[str],
        predictions: List[int],
        ground_truth: List[int] = None
    ) -> Dict[str, Any]:
        """
        Compute statistics from predictions.
        
        Args:
            problem_ids: List of problem IDs
            predictions: List of predictions
            ground_truth: Optional list of ground truth answers
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_problems": len(problem_ids),
            "predictions_made": sum(1 for p in predictions if p is not None),
            "failed_predictions": sum(1 for p in predictions if p is None),
        }
        
        # Add accuracy if ground truth provided
        if ground_truth is not None:
            if len(ground_truth) == len(predictions):
                correct = sum(1 for p, g in zip(predictions, ground_truth) 
                            if p == g and p is not None)
                stats["correct"] = correct
                stats["accuracy"] = correct / len(predictions) if len(predictions) > 0 else 0
            else:
                logger.warning("ground_truth length doesn't match predictions length")
        
        return stats
    
    
    @staticmethod
    def generate_report(
        problem_ids: List[str],
        predictions: List[int],
        reasoning_list: List[str] = None,
        ground_truth: List[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report of results.
        
        Args:
            problem_ids: List of problem IDs
            predictions: List of predictions
            reasoning_list: Optional list of reasoning steps
            ground_truth: Optional list of ground truth answers
            
        Returns:
            Dictionary containing detailed report
        """
        stats = ResultsAggregator.compute_statistics(
            problem_ids, predictions, ground_truth
        )
        
        report = {
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        for idx, (pid, pred) in enumerate(zip(problem_ids, predictions)):
            result = {
                "problem_id": pid,
                "prediction": pred,
            }
            
            if reasoning_list and idx < len(reasoning_list):
                result["reasoning"] = reasoning_list[idx]
            
            if ground_truth and idx < len(ground_truth):
                result["ground_truth"] = ground_truth[idx]
                result["correct"] = (pred == ground_truth[idx])
            
            report["results"].append(result)
        
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test submission formatting
    formatter = SubmissionFormatter()
    problem_ids = ["P1", "P2", "P3"]
    predictions = [42, 17, 100]
    
    formatter.save_submission_csv(problem_ids, predictions)
    
    # Test reasoning logging
    logger_inst = ReasoningLogger()
    reasoning = [
        "Step 1: 2 + 3 = 5\nStep 2: 5 * 5 = 25\nAnswer: 25",
        "Step 1: Solve 2x + 5 = 13\nx = 4\nAnswer: 4",
        "Step 1: 10 * 10 = 100\nAnswer: 100"
    ]
    
    logger_inst.save_reasoning_steps(problem_ids, reasoning, predictions)
    
    # Test statistics
    stats = ResultsAggregator.compute_statistics(
        problem_ids,
        predictions,
        ground_truth=[42, 17, 100]
    )
    print("Statistics:", stats)
