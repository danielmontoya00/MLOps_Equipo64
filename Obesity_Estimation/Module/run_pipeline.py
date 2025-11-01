"""
Machine Learning Pipeline Orchestrator
Executes data processing, model training, and evaluation steps in sequence.
"""
import sys
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
import logging


class PipelineStep(Enum):
    """Enumeration of pipeline steps."""
    DATA_PROCESSING = "notebooks/make_dataset.py"
    TRAINING = "models/train_model.py"
    EVALUATION = "models/evaluate_model.py"


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    steps: List[PipelineStep] = field(default_factory=lambda: [
        PipelineStep.DATA_PROCESSING,
        PipelineStep.TRAINING,
        PipelineStep.EVALUATION
    ])
    python_executable: str = sys.executable
    stop_on_error: bool = True
    log_output: bool = True
    log_file: Optional[Path] = None
    timeout: Optional[int] = None  # seconds, None = no timeout
    capture_output: bool = False  # If True, captures stdout/stderr


@dataclass
class StepResult:
    """Result of a pipeline step execution."""
    step: PipelineStep
    success: bool
    return_code: int
    duration: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class PipelineRunner:
    """Orchestrates the execution of the ML pipeline."""
    
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.results: List[StepResult] = []
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = logging.INFO
        
        if self.config.log_file:
            self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.config.log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(
                level=log_level,
                format=log_format,
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        
        self.logger = logging.getLogger(__name__)
    
    def _validate_step(self, step: PipelineStep) -> bool:
        """Validate that a pipeline step file exists.
        
        Args:
            step: Pipeline step to validate
            
        Returns:
            True if step file exists, False otherwise
        """
        step_path = Path(step.value)
        if not step_path.exists():
            self.logger.error(f"Script not found: {step_path}")
            return False
        return True
    
    def _execute_step(self, step: PipelineStep) -> StepResult:
        """Execute a single pipeline step.
        
        Args:
            step: Pipeline step to execute
            
        Returns:
            StepResult containing execution details
        """
        command = [self.config.python_executable, step.value]
        self.logger.info(f"▶️  EXECUTING STEP: {step.name} ({step.value})")
        
        start_time = time.time()
        
        try:
            if self.config.capture_output:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                stdout = result.stdout
                stderr = result.stderr
                
                # Log captured output
                if stdout:
                    self.logger.info(f"Output from {step.name}:\n{stdout}")
                if stderr:
                    self.logger.warning(f"Errors from {step.name}:\n{stderr}")
            else:
                result = subprocess.run(
                    command,
                    timeout=self.config.timeout
                )
                stdout = None
                stderr = None
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                self.logger.info(
                    f"✅ Step {step.name} completed successfully "
                    f"in {duration:.2f}s"
                )
            else:
                self.logger.error(
                    f"❌ Step {step.name} failed with return code "
                    f"{result.returncode}"
                )
            
            return StepResult(
                step=step,
                success=success,
                return_code=result.returncode,
                duration=duration,
                stdout=stdout,
                stderr=stderr
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Step {step.name} timed out after {self.config.timeout}s"
            self.logger.error(f"⏱️  {error_msg}")
            
            return StepResult(
                step=step,
                success=False,
                return_code=-1,
                duration=duration,
                error_message=error_msg
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error in {step.name}: {str(e)}"
            self.logger.error(f"💥 {error_msg}")
            
            return StepResult(
                step=step,
                success=False,
                return_code=-1,
                duration=duration,
                error_message=error_msg
            )
    
    def run(self) -> bool:
        """Execute the complete pipeline.
        
        Returns:
            True if all steps succeeded, False otherwise
        """
        self.logger.info("🚀 Starting Machine Learning Pipeline...")
        self.logger.info(f"Python executable: {self.config.python_executable}")
        self.logger.info(f"Steps to execute: {len(self.config.steps)}")
        
        pipeline_start = time.time()
        
        # Validate all steps first
        for step in self.config.steps:
            if not self._validate_step(step):
                self.logger.error("❌ Pipeline validation failed")
                return False
        
        # Execute steps
        for i, step in enumerate(self.config.steps, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Step {i}/{len(self.config.steps)}: {step.name}")
            self.logger.info(f"{'='*60}")
            
            result = self._execute_step(step)
            self.results.append(result)
            
            if not result.success and self.config.stop_on_error:
                self.logger.error(
                    f"❌ Pipeline aborted due to failure in step: {step.name}"
                )
                self._print_summary(success=False)
                return False
        
        pipeline_duration = time.time() - pipeline_start
        self.logger.info(f"\n{'='*60}")
        self.logger.info(
            f"✅ Pipeline completed successfully in {pipeline_duration:.2f}s!"
        )
        self.logger.info(f"{'='*60}")
        
        self._print_summary(success=True)
        return True
    
    def _print_summary(self, success: bool) -> None:
        """Print a summary of pipeline execution.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        
        total_duration = sum(r.duration for r in self.results)
        
        for result in self.results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            self.logger.info(
                f"{status} | {result.step.name:20} | "
                f"{result.duration:6.2f}s | "
                f"Exit Code: {result.return_code}"
            )
            if result.error_message:
                self.logger.info(f"         Error: {result.error_message}")
        
        self.logger.info("-" * 60)
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        self.logger.info(
            f"Success Rate: {sum(r.success for r in self.results)}/"
            f"{len(self.results)}"
        )
        self.logger.info("="*60 + "\n")
    
    def get_results(self) -> List[StepResult]:
        """Get the results of all executed steps.
        
        Returns:
            List of StepResult objects
        """
        return self.results


def main():
    """Main entry point for the pipeline runner."""
    # Create configuration
    config = PipelineConfig(
        stop_on_error=True,
        log_output=True,
        # Uncomment to save logs to file:
        # log_file=Path("logs/pipeline.log"),
        # Uncomment to set timeout (in seconds):
        # timeout=3600,
        # Uncomment to capture and log script outputs:
        # capture_output=True
    )
    
    # Run pipeline
    runner = PipelineRunner(config)
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()