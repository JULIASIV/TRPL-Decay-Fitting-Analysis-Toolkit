import numpy as np
from scipy.optimize import curve_fit
import lmfit
from typing import Dict, Tuple, Optional, Union
import warnings

class TRPLFitter:
    """Fit multi-exponential decays to TRPL data"""
    
    @staticmethod
    def monoexponential_model(t: np.ndarray, A: float, tau: float, offset: float = 0) -> np.ndarray:
        return A * np.exp(-t / tau) + offset
    
    @staticmethod
    def biexponential_model(t: np.ndarray, A1: float, tau1: float, 
                            A2: float, tau2: float, offset: float = 0) -> np.ndarray:
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + offset
    
    @staticmethod
    def triexponential_model(t: np.ndarray, A1: float, tau1: float, 
                             A2: float, tau2: float, A3: float, 
                             tau3: float, offset: float = 0) -> np.ndarray:
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3) + offset
    
    def fit_scipy(self, t: np.ndarray, y: np.ndarray, model: str = 'biexponential', 
                  p0: Optional[list] = None, bounds: Optional[tuple] = None, **kwargs) -> Dict:
        """Fit using SciPy's curve_fit"""
        
        if model == 'monoexponential':
            model_func = self.monoexponential_model
            if p0 is None:
                p0 = [np.max(y), 10, np.min(y)]  # A, tau, offset
        elif model == 'biexponential':
            model_func = self.biexponential_model
            if p0 is None:
                p0 = [np.max(y)*0.7, 1, np.max(y)*0.3, 10, np.min(y)]
        elif model == 'triexponential':
            model_func = self.triexponential_model
            if p0 is None:
                p0 = [np.max(y)*0.5, 0.5, np.max(y)*0.3, 5, np.max(y)*0.2, 50, np.min(y)]
        else:
            raise ValueError(f"Model {model} not supported")
        
        try:
            popt, pcov = curve_fit(model_func, t, y, p0=p0, bounds=bounds, **kwargs)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else None
            
            # Calculate R-squared
            residuals = y - model_func(t, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'parameters': popt,
                'covariance': pcov,
                'errors': perr,
                'r_squared': r_squared,
                'model': model,
                'function': model_func
            }
        except Exception as e:
            warnings.warn(f"Fit failed: {e}")
            return None
    
    def fit_lmfit(self, t: np.ndarray, y: np.ndarray, model: str = 'biexponential', 
                  initial_params: Optional[Dict] = None, weights: Optional[np.ndarray] = None) -> lmfit.model.ModelResult:
        """Fit using LMFit with more robust parameter handling"""
        
        if model == 'monoexponential':
            model_lm = lmfit.Model(self.monoexponential_model)
            params = lmfit.Parameters()
            params.add('A', value=np.max(y), min=0)
            params.add('tau', value=10, min=0)
            params.add('offset', value=np.min(y))
            
        elif model == 'biexponential':
            model_lm = lmfit.Model(self.biexponential_model)
            params = lmfit.Parameters()
            params.add('A1', value=np.max(y)*0.7, min=0)
            params.add('tau1', value=1, min=0)
            params.add('A2', value=np.max(y)*0.3, min=0)
            params.add('tau2', value=10, min=0)
            params.add('offset', value=np.min(y))
            
        elif model == 'triexponential':
            model_lm = lmfit.Model(self.triexponential_model)
            params = lmfit.Parameters()
            params.add('A1', value=np.max(y)*0.5, min=0)
            params.add('tau1', value=0.5, min=0)
            params.add('A2', value=np.max(y)*0.3, min=0)
            params.add('tau2', value=5, min=0)
            params.add('A3', value=np.max(y)*0.2, min=0)
            params.add('tau3', value=50, min=0)
            params.add('offset', value=np.min(y))
        
        # Update with user-provided initial parameters
        if initial_params:
            for key, value in initial_params.items():
                if key in params:
                    params[key].value = value
        
        # Perform fit
        if weights is not None:
            result = model_lm.fit(y, params, t=t, weights=weights)
        else:
            result = model_lm.fit(y, params, t=t)
        
        return result
    
    def calculate_average_lifetime(self, fit_result: Union[Dict, lmfit.model.ModelResult]) -> float:
        """Calculate amplitude-weighted average lifetime"""
        
        if isinstance(fit_result, lmfit.model.ModelResult):
            params = fit_result.params
            if 'A3' in params:  # Triexponential
                A1, tau1 = params['A1'].value, params['tau1'].value
                A2, tau2 = params['A2'].value, params['tau2'].value
                A3, tau3 = params['A3'].value, params['tau3'].value
                total_A = A1 + A2 + A3
                return (A1*tau1 + A2*tau2 + A3*tau3) / total_A
            elif 'A2' in params:  # Biexponential
                A1, tau1 = params['A1'].value, params['tau1'].value
                A2, tau2 = params['A2'].value, params['tau2'].value
                return (A1*tau1 + A2*tau2) / (A1 + A2)
            else:  # Monoexponential
                return params['tau'].value
        else:
            # Handle dictionary format
            params = fit_result['parameters']
            if len(params) == 7:  # Triexponential
                A1, tau1, A2, tau2, A3, tau3, _ = params
                total_A = A1 + A2 + A3
                return (A1*tau1 + A2*tau2 + A3*tau3) / total_A
            elif len(params) == 5:  # Biexponential
                A1, tau1, A2, tau2, _ = params
                return (A1*tau1 + A2*tau2) / (A1 + A2)
            else:  # Monoexponential
                return params[1]