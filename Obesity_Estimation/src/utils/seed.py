"""
Utility module for setting random seeds across all libraries.
Ensures reproducibility of experiments and model training.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and environment variables.
    
    This ensures reproducibility across:
    - Python's random module
    - NumPy random operations
    - Scikit-learn (via NumPy)
    - Other libraries that respect these seeds
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Set PYTHONHASHSEED for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🌱 Random seeds set to {seed}")


def get_default_seed() -> int:
    """
    Get the default seed value from environment or return default.
    
    Returns:
        Seed value from RANDOM_SEED env var or 42
    """
    return int(os.getenv('RANDOM_SEED', '42'))
