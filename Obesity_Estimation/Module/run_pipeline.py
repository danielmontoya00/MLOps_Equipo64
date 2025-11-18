"""
Machine Learning Pipeline Orchestrator
Executes data processing, model training, and evaluation steps in sequence.
"""
import sys
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from enum import Enum
import logging
import pandas as pd

# ----------------------
# PASO 0: PREPROCESAMIENTO
# ----------------------
def create_interim_data(raw_path="data/raw/obesity_estimation_final.csv",
                        interim_path="data/interim/interim_data.csv"):
    raw_df = pd.read_csv(raw_path)
    interim_df = raw_df.copy()
    interim_df["age_scaled"] = (interim_df["Age"] - interim_df["Age"].mean()) / interim_df["Age"].std()
    interim_df["height_m"] = interim_df["Height"] / 100
    interim_df["weight_kg"] = interim_df["Weight"]
    interim_df["bmi"] = interim_df["weight_kg"] / interim_df["height_m"]**2
    interim_df["family_history_bool"] = interim_df["family_history_with_overweight"].map({"yes":1,"no":0})
    interim_df["target"] = interim_df["NObeyesdad"].map({
        "Insufficient_Weight":0,
        "Normal_Weight":1,
        "Overweight_Level_I":2,
        "Overweight_Level_II":3,
        "Obesity_Type_I":4,
        "Obesity_Type_II":5,
        "Obesity_Type_III":6
    })
    Path(interim_path).parent.mkdir(parents=True, exist_ok=True)
    interim_df.to_csv(interim_path, index=False)
    return interim_df

def create_processed_data(interim_path="data/interim/interim_data.csv",
                          processed_path="data/processed/data_processed.csv"):
    df = pd.read_csv(interim_path)
    processed_df = df.drop_duplicates().dropna()
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    return processed_df

# ----------------------
# PIPELINE
# ----------------------
class PipelineStep(Enum):
    """Enumeration of pipeline steps."""
    DATA_PREPROCESSING = "Module/preprocess_data.py"  # <- nuevo step
    DATA_PROCESSING = "notebooks/make_dataset.py"
    TRAINING = "models/train_model.py"
    EVALUATION = "models/evaluate_model.py"


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    steps: List[PipelineStep] = field(default_factory=lambda: [
        PipelineStep.DATA_PREPROCESSING,  # ahora primero
        PipelineStep.DATA_PROCESSING,
        PipelineStep.TRAINING,
        PipelineStep.EVALUATION
    ])
    python_executable: str = sys.executable
    stop_on_error: bool = True
    log_output: bool = True
    log_file: Optional[Path] = Path("logs/pipeline.log")
    timeout: Optional[int] = None  # seconds, None = no timeout
    capture_output: bool = True  # Captura stdout/stderr


@dataclass
class StepResult:
    step: PipelineStep
    success: bool
    return_code: int
    duration: float
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class PipelineRunner:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.results: List[StepResult] = []
        self._setup_logging()

    def _setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_level = logging.INFO
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _validate_step(self, step: PipelineStep) -> bool:
        step_path = Path(step.value)
        if not step_path.exists():
            self.logger.error(f"Script not found: {step_path}")
            return False
        return True

    def _execute_step(self, step: PipelineStep) -> StepResult:
        command = [self.config.python_executable, str(Path(step.value).resolve())]
        self.logger.info(f"▶️  EXECUTING STEP: {step.name} ({step.value})")
        start_time = time.time()

        # Definir la raíz del proyecto (dos niveles arriba de Module/)
        project_root = Path(__file__).resolve().parent.parent
        
        try:   
            result = subprocess.run(
               command, 
               capture_output=True, 
               text=True, 
               timeout=self.config.timeout, 
               cwd=project_root
             )
            stdout = result.stdout
            stderr = result.stderr

            if stdout:
                    self.logger.info(stdout)
            if stderr:
                    self.logger.warning(stderr)
            
            duration = time.time() - start_time
            success = result.returncode == 0

            return StepResult(
               step=step,
               success=result.returncode == 0,
               return_code=result.returncode,
               duration=time.time() - start_time,
               stdout=stdout,
               stderr=stderr
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            error_msg = f"Step {step.name} timed out after {self.config.timeout}s"
            self.logger.error(error_msg)
            return StepResult(step, False, -1, duration, error_message=error_msg)
        
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Unexpected error in {step.name}: {str(e)}"
            self.logger.error(error_msg)
            return StepResult(step, False, -1, duration, error_message=error_msg)

    def run(self) -> bool:
        self.logger.info("🚀 Starting Machine Learning Pipeline...")
        for step in self.config.steps:
            if not self._validate_step(step):
                self.logger.error("❌ Pipeline validation failed")
                return False

        for step in self.config.steps:
            result = self._execute_step(step)
            self.results.append(result)
            if not result.success and self.config.stop_on_error:
                self.logger.error(f"❌ Pipeline aborted due to failure in step: {step.name}")
                self._print_summary()
                return False

        self._print_summary()
        return True

    def _print_summary(self):
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        for r in self.results:
            status = "✅ SUCCESS" if r.success else "❌ FAILED"
            self.logger.info(f"{status} | {r.step.name} | {r.duration:.2f}s | Exit Code: {r.return_code}")
            if r.error_message:
                self.logger.info(f"Error: {r.error_message}")
        self.logger.info("="*60 + "\n")


def main():
    config = PipelineConfig()
    runner = PipelineRunner(config)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
