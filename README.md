# TRPL Analyzer

A Python tool for fitting multi-exponential decays to Time-Resolved Photoluminescence (TRPL) data.

## Features

- **Simulated Data Generation**: Create realistic TRPL decays with configurable noise
- **Multi-exponential Fitting**: Fit mono-, bi-, and tri-exponential decay models
- **Multiple Fitting Backends**: Support for both SciPy and LMFit
- **Publication-quality Plots**: Customizable matplotlib styles for scientific publications
- **Real Data Support**: Load data from CSV, Excel, or text files
- **Parameter Extraction**: Calculate lifetimes, amplitudes, and average lifetimes

## Installation

```bash
pip install -e .