import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from typing import Optional, Tuple, Dict, Any
import seaborn as sns

class TRPLPlotter:
    """Create publication-quality TRPL plots"""
    
    def __init__(self, style: str = 'default'):
        """Initialize with specific style settings"""
        if style == 'publication':
            self._set_publication_style()
        elif style == 'presentation':
            self._set_presentation_style()
        else:
            self._set_default_style()
    
    def _set_publication_style(self):
        """Set publication-quality plotting parameters"""
        rcParams['figure.figsize'] = (8, 6)
        rcParams['font.size'] = 12
        rcParams['axes.labelsize'] = 14
        rcParams['axes.titlesize'] = 16
        rcParams['xtick.labelsize'] = 12
        rcParams['ytick.labelsize'] = 12
        rcParams['legend.fontsize'] = 12
        rcParams['lines.linewidth'] = 2
        rcParams['axes.linewidth'] = 1.5
        rcParams['xtick.major.width'] = 1.5
        rcParams['ytick.major.width'] = 1.5
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        rcParams['mathtext.fontset'] = 'stix'
        
    def _set_presentation_style(self):
        """Set presentation-style plotting parameters"""
        rcParams['figure.figsize'] = (10, 7)
        rcParams['font.size'] = 14
        rcParams['axes.labelsize'] = 16
        rcParams['axes.titlesize'] = 18
        rcParams['legend.fontsize'] = 14
        rcParams['lines.linewidth'] = 3
        
    def _set_default_style(self):
        """Set default matplotlib style"""
        pass
    
    def plot_decay(self, t: np.ndarray, decay: np.ndarray, 
                   fit_result: Optional[Dict] = None,
                   log_scale: bool = True,
                   title: str = 'TRPL Decay',
                   xlabel: str = 'Time (ns)',
                   ylabel: str = 'Intensity (a.u.)',
                   save_path: Optional[str] = None,
                   show_components: bool = False,
                   **kwargs) -> plt.Figure:
        """Plot TRPL decay data with optional fit"""
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
        
        # Plot raw data
        ax.plot(t, decay, 'o', markersize=4, alpha=0.7, 
                label='Data', color='blue', markeredgecolor='black')
        
        # Plot fit if provided
        if fit_result is not None:
            if 'function' in fit_result:
                # SciPy fit result
                fit_curve = fit_result['function'](t, *fit_result['parameters'])
                ax.plot(t, fit_curve, '-', linewidth=3, 
                        label='Fit', color='red', alpha=0.8)
                
                # Plot individual components for multi-exponential fits
                if show_components and len(fit_result['parameters']) > 3:
                    self._plot_components(ax, t, fit_result)
                
                # Add fit info to plot
                fit_info = self._create_fit_info_string(fit_result)
                ax.text(0.02, 0.98, fit_info, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set scales
        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Labels and formatting
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_components(self, ax, t, fit_result):
        """Plot individual exponential components"""
        params = fit_result['parameters']
        if len(params) == 5:  # Biexponential
            A1, tau1, A2, tau2, offset = params
            comp1 = A1 * np.exp(-t / tau1) + offset/2
            comp2 = A2 * np.exp(-t / tau2) + offset/2
            ax.plot(t, comp1, '--', label=f'τ₁ = {tau1:.2f} ns', alpha=0.6)
            ax.plot(t, comp2, '--', label=f'τ₂ = {tau2:.2f} ns', alpha=0.6)
        elif len(params) == 7:  # Triexponential
            A1, tau1, A2, tau2, A3, tau3, offset = params
            comp1 = A1 * np.exp(-t / tau1) + offset/3
            comp2 = A2 * np.exp(-t / tau2) + offset/3
            comp3 = A3 * np.exp(-t / tau3) + offset/3
            ax.plot(t, comp1, '--', label=f'τ₁ = {tau1:.2f} ns', alpha=0.6)
            ax.plot(t, comp2, '--', label=f'τ₂ = {tau2:.2f} ns', alpha=0.6)
            ax.plot(t, comp3, '--', label=f'τ₂ = {tau3:.2f} ns', alpha=0.6)
    
    def _create_fit_info_string(self, fit_result: Dict) -> str:
        """Create formatted string with fit results"""
        params = fit_result['parameters']
        errors = fit_result.get('errors', [None]*len(params))
        
        info = f"Fit Results:\n"
        info += f"Model: {fit_result['model']}\n"
        
        if len(params) == 3:  # Monoexponential
            A, tau, offset = params
            info += f"τ = {tau:.3f} ns\n"
            info += f"A = {A:.3f}\n"
            info += f"R² = {fit_result.get('r_squared', 0):.4f}"
            
        elif len(params) == 5:  # Biexponential
            A1, tau1, A2, tau2, offset = params
            total_A = A1 + A2
            info += f"τ₁ = {tau1:.3f} ns ({(A1/total_A*100):.1f}%)\n"
            info += f"τ₂ = {tau2:.3f} ns ({(A2/total_A*100):.1f}%)\n"
            info += f"⟨τ⟩ = {(A1*tau1 + A2*tau2)/total_A:.3f} ns\n"
            info += f"R² = {fit_result.get('r_squared', 0):.4f}"
            
        elif len(params) == 7:  # Triexponential
            A1, tau1, A2, tau2, A3, tau3, offset = params
            total_A = A1 + A2 + A3
            info += f"τ₁ = {tau1:.3f} ns ({(A1/total_A*100):.1f}%)\n"
            info += f"τ₂ = {tau2:.3f} ns ({(A2/total_A*100):.1f}%)\n"
            info += f"τ₃ = {tau3:.3f} ns ({(A3/total_A*100):.1f}%)\n"
            info += f"⟨τ⟩ = {(A1*tau1 + A2*tau2 + A3*tau3)/total_A:.3f} ns\n"
            info += f"R² = {fit_result.get('r_squared', 0):.4f}"
        
        return info
    
    def plot_residuals(self, t: np.ndarray, y: np.ndarray, 
                       fit_result: Dict, 
                       title: str = 'Fit Residuals',
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot residuals of the fit"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot data and fit
        fit_curve = fit_result['function'](t, *fit_result['parameters'])
        ax1.plot(t, y, 'o', label='Data', markersize=4, alpha=0.7)
        ax1.plot(t, fit_curve, '-', label='Fit', linewidth=2)
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot residuals
        residuals = y - fit_curve
        ax2.plot(t, residuals, 'o', markersize=4, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Residuals')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig