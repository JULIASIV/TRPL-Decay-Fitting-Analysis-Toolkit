import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import json
import os

class TRPLDataIO:
    """Handle loading and saving of TRPL data"""
    
    @staticmethod
    def load_csv(filepath: str, time_col: str = 'time', 
                 intensity_col: str = 'intensity',
                 delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
        """Load TRPL data from CSV file"""
        try:
            df = pd.read_csv(filepath, delimiter=delimiter)
            t = df[time_col].values
            y = df[intensity_col].values
            return t, y
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    @staticmethod
    def load_txt(filepath: str, skip_rows: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Load TRPL data from text file"""
        try:
            data = np.loadtxt(filepath, skiprows=skip_rows)
            if data.shape[1] >= 2:
                return data[:, 0], data[:, 1]
            else:
                raise ValueError("File must have at least two columns")
        except Exception as e:
            raise ValueError(f"Error loading text file: {e}")
    
    @staticmethod
    def load_excel(filepath: str, sheet_name: str = 0,
                   time_col: str = 'time', 
                   intensity_col: str = 'intensity') -> Tuple[np.ndarray, np.ndarray]:
        """Load TRPL data from Excel file"""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            t = df[time_col].values
            y = df[intensity_col].values
            return t, y
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {e}")
    
    @staticmethod
    def save_fit_results(results: Dict, filepath: str):
        """Save fit results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif hasattr(value, '__call__'):  # Skip functions
                continue
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    @staticmethod
    def save_data_csv(t: np.ndarray, y: np.ndarray, 
                      fit: Optional[np.ndarray] = None,
                      filepath: str = 'trpl_data.csv'):
        """Save data (and optional fit) to CSV"""
        data_dict = {'time': t, 'intensity': y}
        if fit is not None:
            data_dict['fit'] = fit
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)