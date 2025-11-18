"""
Data Drift Simulator

Creates synthetic datasets with various distribution shifts to simulate
real-world drift scenarios:
- Mean shift (covariate drift)
- Missing features
- Seasonal changes
- Label drift
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.seed import set_seed


class DriftType(Enum):
    """Types of data drift to simulate."""
    MEAN_SHIFT = "mean_shift"
    FEATURE_MISSING = "feature_missing"
    SEASONAL = "seasonal"
    LABEL_DRIFT = "label_drift"
    VARIANCE_SHIFT = "variance_shift"
    COMBINED = "combined"


@dataclass
class DriftConfig:
    """Configuration for drift simulation."""
    drift_type: DriftType = DriftType.MEAN_SHIFT
    intensity: float = 0.3  # Intensity of the drift (0-1)
    affected_features: Optional[list] = None  # Features to affect, None = all
    missing_probability: float = 0.2  # For FEATURE_MISSING type
    seasonal_amplitude: float = 0.5  # For SEASONAL type
    random_state: int = 42


class DriftSimulator:
    """Simulates various types of data drift on datasets."""
    
    def __init__(self, config: DriftConfig = DriftConfig()):
        """
        Initialize the drift simulator.
        
        Args:
            config: Configuration for drift simulation
        """
        self.config = config
        # Set seed for reproducibility
        set_seed(config.random_state)
        np.random.seed(config.random_state)
        
    def load_baseline_data(
        self, 
        data_dir: Path = Path("data/processed")
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load baseline validation/test data.
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Tuple of (X_test, y_test)
        """
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()
        
        print(f"Loaded baseline data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        return X_test, y_test
    
    def simulate_mean_shift(
        self, 
        X: pd.DataFrame, 
        intensity: float = 0.3
    ) -> pd.DataFrame:
        """
        Simulate covariate drift by shifting feature means.
        
        Args:
            X: Original features
            intensity: Magnitude of mean shift (0-1)
            
        Returns:
            Modified features with shifted means
        """
        X_drift = X.copy()
        
        # Select features to shift
        if self.config.affected_features:
            features = self.config.affected_features
        else:
            # Select numeric features only
            features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply mean shift
        for feature in features:
            if feature in X_drift.columns:
                mean_val = X_drift[feature].mean()
                std_val = X_drift[feature].std()
                # Shift by intensity * std
                shift = intensity * std_val * np.random.choice([-1, 1])
                X_drift[feature] = X_drift[feature] + shift
                
        print(f"Applied mean shift to {len(features)} features with intensity {intensity}")
        return X_drift
    
    def simulate_variance_shift(
        self, 
        X: pd.DataFrame, 
        intensity: float = 0.3
    ) -> pd.DataFrame:
        """
        Simulate drift by changing feature variances.
        
        Args:
            X: Original features
            intensity: Magnitude of variance change (0-1)
            
        Returns:
            Modified features with altered variances
        """
        X_drift = X.copy()
        
        # Select numeric features
        if self.config.affected_features:
            features = self.config.affected_features
        else:
            features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply variance shift
        for feature in features:
            if feature in X_drift.columns:
                mean_val = X_drift[feature].mean()
                std_val = X_drift[feature].std()
                # Scale variance by (1 + intensity)
                scale_factor = 1 + intensity
                X_drift[feature] = mean_val + (X_drift[feature] - mean_val) * scale_factor
                
        print(f"Applied variance shift to {len(features)} features with intensity {intensity}")
        return X_drift
    
    def simulate_feature_missing(
        self, 
        X: pd.DataFrame, 
        missing_prob: float = 0.2
    ) -> pd.DataFrame:
        """
        Simulate missing data in features.
        
        Args:
            X: Original features
            missing_prob: Probability of a value being missing
            
        Returns:
            Modified features with missing values
        """
        X_drift = X.copy()
        
        # Select features to affect
        if self.config.affected_features:
            features = self.config.affected_features
        else:
            # Affect a subset of features
            all_features = X.columns.tolist()
            n_affected = max(1, int(len(all_features) * 0.3))
            features = np.random.choice(all_features, n_affected, replace=False)
        
        # Introduce missing values
        for feature in features:
            if feature in X_drift.columns:
                mask = np.random.random(len(X_drift)) < missing_prob
                X_drift.loc[mask, feature] = np.nan
        
        # Fill NaN with feature mean (simple imputation)
        X_drift = X_drift.fillna(X_drift.mean())
        
        print(f"Simulated missing data in {len(features)} features with probability {missing_prob}")
        return X_drift
    
    def simulate_seasonal_drift(
        self, 
        X: pd.DataFrame, 
        amplitude: float = 0.5
    ) -> pd.DataFrame:
        """
        Simulate seasonal drift with periodic patterns.
        
        Args:
            X: Original features
            amplitude: Amplitude of seasonal effect
            
        Returns:
            Modified features with seasonal patterns
        """
        X_drift = X.copy()
        n_samples = len(X_drift)
        
        # Select numeric features
        if self.config.affected_features:
            features = self.config.affected_features
        else:
            features = X.select_dtypes(include=[np.number]).columns.tolist()[:3]  # Top 3
        
        # Apply sinusoidal seasonal pattern
        time_steps = np.linspace(0, 2 * np.pi, n_samples)
        seasonal_pattern = amplitude * np.sin(time_steps)
        
        for feature in features:
            if feature in X_drift.columns:
                std_val = X_drift[feature].std()
                X_drift[feature] = X_drift[feature] + seasonal_pattern * std_val
                
        print(f"Applied seasonal drift to {len(features)} features with amplitude {amplitude}")
        return X_drift
    
    def simulate_label_drift(
        self, 
        y: pd.Series, 
        shift_prob: float = 0.15
    ) -> pd.Series:
        """
        Simulate concept drift by modifying labels.
        
        Args:
            y: Original labels
            shift_prob: Probability of label change
            
        Returns:
            Modified labels
        """
        y_drift = y.copy()
        unique_labels = np.unique(y)
        
        # Randomly change some labels
        mask = np.random.random(len(y_drift)) < shift_prob
        n_changes = mask.sum()
        
        if n_changes > 0:
            y_drift[mask] = np.random.choice(unique_labels, n_changes)
        
        print(f"Modified {n_changes} labels ({shift_prob*100:.1f}% of data)")
        return y_drift
    
    def apply_drift(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Apply the configured drift type to the data.
        
        Args:
            X: Original features
            y: Original labels (optional)
            
        Returns:
            Tuple of (X_drifted, y_drifted)
        """
        print(f"\n=== Simulating {self.config.drift_type.value} drift ===")
        
        if self.config.drift_type == DriftType.MEAN_SHIFT:
            X_drift = self.simulate_mean_shift(X, self.config.intensity)
            y_drift = y
            
        elif self.config.drift_type == DriftType.VARIANCE_SHIFT:
            X_drift = self.simulate_variance_shift(X, self.config.intensity)
            y_drift = y
            
        elif self.config.drift_type == DriftType.FEATURE_MISSING:
            X_drift = self.simulate_feature_missing(X, self.config.missing_probability)
            y_drift = y
            
        elif self.config.drift_type == DriftType.SEASONAL:
            X_drift = self.simulate_seasonal_drift(X, self.config.seasonal_amplitude)
            y_drift = y
            
        elif self.config.drift_type == DriftType.LABEL_DRIFT:
            X_drift = X.copy()
            y_drift = self.simulate_label_drift(y, self.config.intensity) if y is not None else None
            
        elif self.config.drift_type == DriftType.COMBINED:
            # Apply multiple drift types
            X_drift = self.simulate_mean_shift(X, self.config.intensity * 0.5)
            X_drift = self.simulate_variance_shift(X_drift, self.config.intensity * 0.3)
            y_drift = y
            
        else:
            X_drift = X.copy()
            y_drift = y
        
        return X_drift, y_drift
    
    def generate_drifted_dataset(
        self,
        output_dir: Path = Path("data/monitoring"),
        suffix: str = "drifted"
    ) -> Dict[str, Path]:
        """
        Generate and save a drifted dataset.
        
        Args:
            output_dir: Directory to save the drifted data
            suffix: Suffix for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        # Load baseline data
        X_baseline, y_baseline = self.load_baseline_data()
        
        # Apply drift
        X_drift, y_drift = self.apply_drift(X_baseline, y_baseline)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save drifted data
        x_path = output_dir / f"X_{suffix}.csv"
        y_path = output_dir / f"y_{suffix}.csv"
        
        X_drift.to_csv(x_path, index=False)
        if y_drift is not None:
            pd.DataFrame(y_drift, columns=['NObeyesdad']).to_csv(y_path, index=False)
        
        print(f"\n✅ Drifted dataset saved:")
        print(f"   Features: {x_path}")
        print(f"   Labels: {y_path}")
        
        return {
            'X_path': x_path,
            'y_path': y_path,
            'drift_type': self.config.drift_type.value,
            'intensity': self.config.intensity
        }


def main():
    """Demo: Generate multiple drift scenarios."""
    print("=== Data Drift Simulator Demo ===\n")
    
    scenarios = [
        ("mean_shift", DriftType.MEAN_SHIFT, 0.3),
        ("variance_shift", DriftType.VARIANCE_SHIFT, 0.4),
        ("seasonal", DriftType.SEASONAL, 0.5),
        ("combined", DriftType.COMBINED, 0.3),
    ]
    
    for name, drift_type, intensity in scenarios:
        config = DriftConfig(
            drift_type=drift_type,
            intensity=intensity,
            random_state=42
        )
        
        simulator = DriftSimulator(config)
        simulator.generate_drifted_dataset(suffix=name)
        print()


if __name__ == "__main__":
    main()
