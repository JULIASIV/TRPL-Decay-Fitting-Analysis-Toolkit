import numpy as np
from typing import Tuple, List, Optional

class TRPLDataGenerator:
    """Generate simulated TRPL decay curves with noise"""
    
    @staticmethod
    def monoexponential(t: np.ndarray, A: float, tau: float, offset: float = 0) -> np.ndarray:
        """Single exponential decay"""
        return A * np.exp(-t / tau) + offset
    
    @staticmethod
    def biexponential(t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float, offset: float = 0) -> np.ndarray:
        """Bi-exponential decay"""
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + offset
    
    @staticmethod
    def triexponential(t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float, 
                       A3: float, tau3: float, offset: float = 0) -> np.ndarray:
        """Tri-exponential decay"""
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3) + offset
    
    @classmethod
    def generate_decay(cls, t: np.ndarray, params: dict, noise_level: float = 0.01, 
                       model: str = 'biexponential') -> Tuple[np.ndarray, np.ndarray]:
        """Generate decay curve with optional noise"""
        
        if model == 'monoexponential':
            decay = cls.monoexponential(t, **params)
        elif model == 'biexponential':
            decay = cls.biexponential(t, **params)
        elif model == 'triexponential':
            decay = cls.triexponential(t, **params)
        else:
            raise ValueError(f"Model {model} not supported")
        
        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * np.max(decay), len(decay))
            decay_noisy = decay + noise
        else:
            decay_noisy = decay.copy()
        
        return decay, decay_noisy
    
    @classmethod
    def create_time_axis(cls, num_points: int = 1000, t_max: float = 100) -> np.ndarray:
        """Create logarithmically spaced time axis (typical for TRPL)"""
        return np.logspace(-2, np.log10(t_max), num_points)