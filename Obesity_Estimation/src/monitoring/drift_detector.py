"""
Data Drift Detector

Statistical tests and metrics to detect distribution shifts between
baseline and monitoring datasets using:
- Kolmogorov-Smirnov test for numerical features
- Chi-square test for categorical features
- Population Stability Index (PSI)
- Jensen-Shannon divergence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    ks_pvalue: float = 0.05  # Kolmogorov-Smirnov p-value threshold
    psi_threshold: float = 0.2  # PSI threshold (>0.2 = significant drift)
    chi2_pvalue: float = 0.05  # Chi-square p-value threshold
    js_divergence: float = 0.1  # Jensen-Shannon divergence threshold


@dataclass
class FeatureDriftResult:
    """Results of drift detection for a single feature."""
    feature_name: str
    is_drifted: bool
    test_statistic: float
    p_value: Optional[float] = None
    psi_score: Optional[float] = None
    drift_magnitude: float = 0.0
    test_type: str = "unknown"


@dataclass
class DriftReport:
    """Complete drift detection report."""
    n_features: int
    n_drifted_features: int
    drift_score: float  # Overall drift score (0-1)
    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    alert_level: str = "NONE"  # NONE, LOW, MEDIUM, HIGH, CRITICAL
    
    def get_drifted_features(self) -> List[str]:
        """Get list of features with detected drift."""
        return [
            name for name, result in self.feature_results.items()
            if result.is_drifted
        ]
    
    def summary(self) -> str:
        """Generate a text summary of the drift report."""
        drifted = self.get_drifted_features()
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    DRIFT DETECTION REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

Overall Drift Score: {self.drift_score:.3f}
Alert Level: {self.alert_level}

Features Analyzed: {self.n_features}
Features with Drift: {self.n_drifted_features} ({self.n_drifted_features/self.n_features*100:.1f}%)

"""
        if drifted:
            summary += "⚠️  Drifted Features:\n"
            for feat in drifted:
                result = self.feature_results[feat]
                summary += f"  • {feat}: "
                if result.p_value is not None:
                    summary += f"p-value={result.p_value:.4f}, "
                if result.psi_score is not None:
                    summary += f"PSI={result.psi_score:.4f}, "
                summary += f"magnitude={result.drift_magnitude:.3f}\n"
        else:
            summary += "✅ No significant drift detected\n"
        
        return summary


class DriftDetector:
    """Detects data drift using statistical tests."""
    
    def __init__(self, thresholds: DriftThresholds = DriftThresholds()):
        """
        Initialize drift detector.
        
        Args:
            thresholds: Thresholds for drift detection
        """
        self.thresholds = thresholds
        
    def calculate_psi(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Small change
        - PSI ≥ 0.2: Significant change (drift detected)
        
        Args:
            baseline: Baseline feature values
            current: Current/monitoring feature values
            bins: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Handle edge cases
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        
        # Create bins based on baseline distribution
        try:
            breakpoints = np.histogram(baseline, bins=bins)[1]
        except:
            return 0.0
        
        # Count samples in each bin
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]
        
        # Calculate proportions (add small epsilon to avoid log(0))
        epsilon = 1e-10
        baseline_percents = (baseline_counts + epsilon) / (len(baseline) + epsilon * bins)
        current_percents = (current_counts + epsilon) / (len(current) + epsilon * bins)
        
        # Calculate PSI
        psi = np.sum((current_percents - baseline_percents) * 
                     np.log(current_percents / baseline_percents))
        
        return abs(psi)
    
    def jensen_shannon_divergence(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between distributions.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            bins: Number of bins for discretization
            
        Returns:
            JS divergence score (0-1)
        """
        # Create histograms
        try:
            breakpoints = np.linspace(
                min(baseline.min(), current.min()),
                max(baseline.max(), current.max()),
                bins + 1
            )
        except:
            return 0.0
        
        baseline_hist, _ = np.histogram(baseline, bins=breakpoints, density=True)
        current_hist, _ = np.histogram(current, bins=breakpoints, density=True)
        
        # Normalize to probability distributions
        epsilon = 1e-10
        p = baseline_hist + epsilon
        q = current_hist + epsilon
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate JS divergence
        m = 0.5 * (p + q)
        divergence = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
        
        return divergence
    
    def kolmogorov_smirnov_test(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for numerical features.
        
        Tests if two samples come from the same distribution.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            
        Returns:
            Tuple of (statistic, p_value)
        """
        try:
            statistic, p_value = stats.ks_2samp(baseline, current)
            return statistic, p_value
        except:
            return 0.0, 1.0
    
    def chi_square_test(
        self, 
        baseline: np.ndarray, 
        current: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, float]:
        """
        Perform Chi-square test for categorical or binned numerical features.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            bins: Number of bins for numerical features
            
        Returns:
            Tuple of (statistic, p_value)
        """
        try:
            # Create bins
            all_values = np.concatenate([baseline, current])
            breakpoints = np.histogram(all_values, bins=bins)[1]
            
            # Get counts
            baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
            current_counts = np.histogram(current, bins=breakpoints)[0]
            
            # Perform chi-square test
            statistic, p_value = stats.chisquare(
                current_counts + 1,  # Add 1 to avoid zeros
                baseline_counts + 1
            )
            return statistic, p_value
        except:
            return 0.0, 1.0
    
    def detect_feature_drift(
        self,
        baseline: pd.Series,
        current: pd.Series,
        feature_name: str
    ) -> FeatureDriftResult:
        """
        Detect drift for a single feature.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            feature_name: Name of the feature
            
        Returns:
            FeatureDriftResult with drift detection results
        """
        # Remove NaN values
        baseline_clean = baseline.dropna().values
        current_clean = current.dropna().values
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return FeatureDriftResult(
                feature_name=feature_name,
                is_drifted=False,
                test_statistic=0.0,
                test_type="skipped"
            )
        
        # Calculate PSI
        psi_score = self.calculate_psi(baseline_clean, current_clean)
        
        # Perform KS test for numerical features
        ks_stat, ks_pvalue = self.kolmogorov_smirnov_test(baseline_clean, current_clean)
        
        # Calculate JS divergence
        js_div = self.jensen_shannon_divergence(baseline_clean, current_clean)
        
        # Determine if drift is detected
        is_drifted = (
            psi_score > self.thresholds.psi_threshold or
            ks_pvalue < self.thresholds.ks_pvalue or
            js_div > self.thresholds.js_divergence
        )
        
        # Calculate combined drift magnitude
        drift_magnitude = (psi_score / self.thresholds.psi_threshold + 
                          js_div / self.thresholds.js_divergence) / 2
        
        return FeatureDriftResult(
            feature_name=feature_name,
            is_drifted=is_drifted,
            test_statistic=ks_stat,
            p_value=ks_pvalue,
            psi_score=psi_score,
            drift_magnitude=drift_magnitude,
            test_type="KS+PSI+JS"
        )
    
    def detect_dataset_drift(
        self,
        X_baseline: pd.DataFrame,
        X_current: pd.DataFrame
    ) -> DriftReport:
        """
        Detect drift across entire dataset.
        
        Args:
            X_baseline: Baseline features
            X_current: Current/monitoring features
            
        Returns:
            DriftReport with comprehensive drift analysis
        """
        print("\n=== Detecting Data Drift ===")
        print(f"Baseline samples: {len(X_baseline)}")
        print(f"Current samples: {len(X_current)}")
        
        feature_results = {}
        
        # Check each feature
        for feature in X_baseline.columns:
            if feature not in X_current.columns:
                warnings.warn(f"Feature {feature} not found in current data, skipping")
                continue
            
            result = self.detect_feature_drift(
                X_baseline[feature],
                X_current[feature],
                feature
            )
            feature_results[feature] = result
        
        # Calculate overall metrics
        n_features = len(feature_results)
        n_drifted = sum(1 for r in feature_results.values() if r.is_drifted)
        drift_ratio = n_drifted / n_features if n_features > 0 else 0
        
        # Average drift magnitude
        avg_magnitude = np.mean([r.drift_magnitude for r in feature_results.values()])
        
        # Overall drift score (0-1)
        drift_score = min(1.0, (drift_ratio * 0.5 + avg_magnitude * 0.5))
        
        # Determine alert level
        if drift_score >= 0.7:
            alert_level = "CRITICAL"
        elif drift_score >= 0.5:
            alert_level = "HIGH"
        elif drift_score >= 0.3:
            alert_level = "MEDIUM"
        elif drift_score >= 0.1:
            alert_level = "LOW"
        else:
            alert_level = "NONE"
        
        report = DriftReport(
            n_features=n_features,
            n_drifted_features=n_drifted,
            drift_score=drift_score,
            feature_results=feature_results,
            alert_level=alert_level
        )
        
        print(f"\n✅ Drift detection complete:")
        print(f"   {n_drifted}/{n_features} features drifted")
        print(f"   Overall drift score: {drift_score:.3f}")
        print(f"   Alert level: {alert_level}")
        
        return report


def main():
    """Demo: Detect drift on simulated data."""
    print("=== Data Drift Detector Demo ===\n")
    
    # Load baseline data
    baseline_path = Path("data/processed")
    X_baseline = pd.read_csv(baseline_path / "X_test.csv")
    
    # Load drifted data (should be generated first)
    drift_path = Path("data/monitoring")
    if not (drift_path / "X_mean_shift.csv").exists():
        print("⚠️  Drifted data not found. Run drift_simulator.py first.")
        return
    
    X_drifted = pd.read_csv(drift_path / "X_mean_shift.csv")
    
    # Detect drift
    detector = DriftDetector()
    report = detector.detect_dataset_drift(X_baseline, X_drifted)
    
    # Print report
    print(report.summary())


if __name__ == "__main__":
    main()
