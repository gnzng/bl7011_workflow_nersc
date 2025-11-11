# BL7011 XPCS Workflow

This repository contains a workflow for X-ray Photon Correlation Spectroscopy (XPCS) data analysis using the BL7011 beamline dataset tools also NERSC with GPU acceleration.

## Overview

The workflow demonstrates how to:
- Load and visualize XPCS data from HDF5 files
- Define regions of interest (ROIs) for analysis
- Compute and plot G2 correlation functions
- Generate and visualize two-time correlation functions

## Requirements (all of that should be preinstalled in the pytorch kernel)

Clone the repository into your personal workspace:
```bash
git clone https://github.com/gnzng/bl7011_workflow_nersc.git
```

uses:
- Python 3.x
- matplotlib
- numpy
- PyTorch (with CUDA)
- h5py

## Overview

1. **GPU Check**: The workflow first verifies CUDA availability for accelerated processing
2. **Data Loading**: Loads XPCS data from HDF5 files with optional plotting and averaging first frames
3. **ROI Definition**: Sets up multiple regions of interest with pixel coordinates, use colors to specify ROIs
4. **Analysis Pipeline**:
    - Visualizes single frames with ROI overlays
    - Computes G2 correlation functions for all defined ROIs
    - Generates two-time correlation analysis
    - Creates comprehensive plots for all results

## Data Structure

The workflow expects HDF5 files containing XPCS data. Replace the path `data/example_data/` with the actual directory.

## ROI Configuration

ROIs are defined as dictionaries with color-coded names and pixel coordinate ranges:
- Format: `"color": [[x_start, x_end], [y_start, y_end]]`
- Multiple ROIs can be analyzed simultaneously

## Output

The workflow generates:
- Single frame visualizations with ROI overlays
- G2 correlation function plots
- Two-time correlation function visualizations


## Quickstart

See also the juptyer notebook as an example: https://github.com/gnzng/bl7011_workflow_nersc/blob/main/bl7011_workflow.ipynb


```python
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from BL7011 import dataset_xpcs # this is where we can call specific things from the beam line, this can be constantly updated
import h5py

# check if gpu is available
print("CUDA available:", t.cuda.is_available())

# replace with the actual directory
filepath = "data/example_data/"


# filename can be the complete name, or a unique identifier in the directory
filename = "1kimgs_83m2"

# Load data
xpcs_set = dataset_xpcs()
xpcs_set.load_file(filepath, filename, plot=True, log_plot=True, average_first_n=10)


xpcs_set.set_rois({
    "red": [[790, 800], [90,100]],
    "yellow": [[500, 550], [300,350]],
    #"blue": [[600, 800], [300,450]],
})
xpcs_set.plot_single_frame_with_roi(log_plot=True)


# compute and plot calculated g2 curves
xpcs_set.compute_g2_for_all_rois()
xpcs_set.plot_g2curve()

# this will take a little longer, calculate and plot twotime correlation function
xpcs_set.compute_twotime_for_all_rois()
xpcs_set.plot_twotime()
```
