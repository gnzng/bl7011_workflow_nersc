import numpy as np
import torch as t
import h5py
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Union
from pathlib import Path

# Configure numpy warnings
np.seterr(divide="ignore")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def g2_fft_t(
    img_stack: np.ndarray, device: str = "cpu", cut_last_ratio: float = 0.5
) -> np.ndarray:
    """
    Compute the g2 correlation function using FFT.

    Parameters
    ----------
    img_stack : numpy.ndarray
        Input image stack with shape (T, H, W).
    device : str, optional
        Device to perform computations on ('cpu' or 'cuda'). Default is 'cpu'.
    cut_last_ratio : float, optional
        Ratio to cut the result length to reduce noise. Default is 0.5.

    Returns
    -------
    numpy.ndarray
        The computed g2 correlation function with reduced noise.
    """
    if len(img_stack.shape) == 2:
        img_stack = img_stack.reshape(*img_stack.shape, 1)

    # Convert to tensor and move to device
    if not isinstance(img_stack, t.Tensor):
        img_stack = t.tensor(img_stack, dtype=t.float32)
    img_stack = img_stack.to(device)

    # FFT calculations
    img_stack_padded = t.cat([img_stack, t.zeros_like(img_stack)], dim=0)
    img_stack_fft = t.fft.fft(img_stack_padded, dim=0)
    numerator_base = t.fft.ifft(img_stack_fft * img_stack_fft.conj(), dim=0)[
        : img_stack.shape[0]
    ].real

    n_elements = t.arange(img_stack.shape[0], device=device) + 1
    n_elements = n_elements.flip(0)
    numerator_base = numerator_base / n_elements.view(-1, 1, 1)

    # Denominator calculations
    lcumsum = t.roll(t.cumsum(img_stack, dim=0), 1, dims=0)
    lcumsum[0, :, :] = 0
    rcumsum = t.roll(t.cumsum(img_stack.flip(0), dim=0), 1, dims=0)
    rcumsum[0, :, :] = 0
    denominator_base = (2 * img_stack.sum(dim=0)) - lcumsum - rcumsum
    n_elements = 2 * img_stack.shape[0] - 2 * t.arange(
        img_stack.shape[0], device=device
    )
    denominator_base = denominator_base / n_elements.view(-1, 1, 1)

    # Calculate final result
    result = numerator_base / denominator_base.pow(2)

    # Remove the second half of the result because of noise
    cut_length = int(len(result) * cut_last_ratio)
    result = result[:cut_length, :, :]

    return result.cpu().numpy()


def twotime(patterns: t.Tensor, device: str = "cpu") -> Optional[t.Tensor]:
    """
    Compute the 2-time correlation function for the given patterns.

    Parameters
    ----------
    patterns : torch.Tensor
        Input tensor with shape (T, H, W) or (T, W).
    device : str, optional
        Device to perform computations on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    torch.Tensor or None
        The computed 2-time correlation function or None if input is invalid.
    """
    if not isinstance(patterns, t.Tensor):
        try:
            patterns = t.tensor(patterns, dtype=t.float32)
        except Exception as e:
            print("Error converting patterns to tensor:", e)
            return None

    # proceed the patterns tensor to device
    patterns = patterns.to(device)

    # Get dimensions
    num_frames, height, width = patterns.shape
    device = patterns.device

    # Flatten spatial dimensions
    patterns_flat = patterns.reshape(num_frames, -1)

    # we only want one ROI for this function, we can iterate over multiple ROIs later
    # Initialize g2 matrix
    g2 = t.zeros(num_frames, num_frames, device=device)

    # Calculate mean intensity at each time
    mean_intensities = patterns_flat.mean(dim=1)  # shape: (time,)

    # Calculate g2 for all time pairs
    for t1 in range(num_frames):
        for t2 in range(num_frames):
            # Calculate <I(t1)*I(t2)> / (<I(t1)>*<I(t2)>)
            numerator = (patterns_flat[t1] * patterns_flat[t2]).mean(axis=0)
            denominator = mean_intensities[t1] * mean_intensities[t2]

            if denominator > 0:
                g2[t1, t2] = numerator / denominator

            # We want to ignore the self correlation, so we set the diagonal to NaN
            if t1 == t2:
                g2[t1, t2] = t.nan

    return g2


def mean_every_n_frames(patterns: t.Tensor, n: int) -> Optional[t.Tensor]:
    """
    Averages every n frames along the first dimension of a tensor.

    Parameters
    ----------
    patterns : torch.Tensor
        The input tensor with shape (frames, height, width) or (frames, width)
    n : int
        The number of frames to average.

    Returns
    -------
    torch.Tensor or None
        A tensor with averaged frames or None if frames not divisible by n.
    """
    patterns_subset = patterns[: patterns.shape[0] // n * n]
    frames = patterns_subset.shape[0]

    if frames % n != 0:
        return None

    if len(patterns_subset.shape) == 3:
        frames, height, width = patterns_subset.shape
        return patterns_subset.reshape(frames // n, n, height, width).mean(dim=1)
    elif len(patterns_subset.shape) == 2:
        frames, width = patterns_subset.shape
        height = 1
        return patterns_subset.reshape(frames // n, n, width, height).mean(dim=1)


# ============================================================================
# DATA LOADER CLASS
# ============================================================================


class AndorDataLoader:
    """Class for loading and processing Andor detector data from HDF5 files."""

    def __init__(
        self,
        filepath: str,
        filename: str,
        norm: str = "mean",
        roi: Optional[Tuple] = None,
    ):
        """
        Initialize the data loader.

        Parameters
        ----------
        filepath : str
            Path to the directory containing the data file
        filename : str
            Name of the data file
        norm : str, optional
            Normalization method ('mean', 'sum', or 'none')
        roi : tuple, optional
            Region of interest as (y0, y1, x0, x1) or 'find' to display image
        """
        self.filepath = Path(filepath)
        self.filename = filename
        self.norm = norm
        self.rois: Dict[str, Tuple] = {}

        # Find and validate file
        self.file_path = self._find_file()
        self.filename_truncated = self.file_path.name

        print(f"Found file: {self.file_path}")

        # Load data
        self._load_data(roi)

    def _find_file(self) -> Path:
        """Find the data file in the specified directory."""
        full_path = self.filepath / self.filename

        if full_path.exists():
            return full_path

        # Search for files containing the filename pattern
        matching_files = [f for f in self.filepath.iterdir() if self.filename in f.name]

        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No file matching pattern '{self.filename}' found in {self.filepath}"
            )
        elif len(matching_files) > 1:
            raise FileNotFoundError(
                f"Multiple files matching pattern '{self.filename}' found: {[f.name for f in matching_files]}"
            )

        return matching_files[0]

    def _load_data(self, roi: Optional[Union[str, Tuple]]):
        """Load data from HDF5 file."""
        with h5py.File(self.file_path, "r") as f:
            if roi == "find":
                self._display_roi_finder(f)
                return

            # Load patterns with optional ROI
            patterns = self._extract_patterns(f, roi)

            # Load metadata
            try:
                self.periods = self._extract_periods(f)
            except KeyError:
                print("Periods data not found in the file, initializing empty array.")
                self.periods = np.array([])
            try:
                self.temps = self._extract_temperatures(f)
            except KeyError:
                print("Temperature data not found in the file, initializing empty array.")
                self.temps = np.array([])

            # Store original data
            self.patterns = patterns.astype(np.float32)
            self.patterns_mean = np.nanmean(patterns, axis=0)
            self.patterns_sum = np.nansum(patterns, axis=0)
            self.original_mean_intensity = np.nanmean(self.patterns, axis=(1, 2))

            # Apply normalization
            self._apply_normalization()

    def _extract_patterns(self, f: h5py.File, roi: Optional[Tuple]) -> np.ndarray:
        """Extract pattern data from HDF5 file."""
        data_path = "entry1/instrument_1/detector_1/data"

        try:
            if roi is not None and roi != "find":
                patterns = np.squeeze(
                    f[data_path][:, :, roi[0] : roi[1], roi[2] : roi[3]]
                )
            else:
                patterns = np.squeeze(f[data_path][:, :, :, :])
        except KeyError:
            patterns = np.squeeze(f[data_path][:, :, :, :])

        return patterns

    def _extract_periods(self, f: h5py.File) -> np.ndarray:
        """Extract period data from HDF5 file."""
        periods = f["entry1/instrument_1/detector_1/period"][()]
        count_time = f["entry1/instrument_1/detector_1/count_time"][()]
        return periods + count_time

    def _extract_temperatures(self, f: h5py.File) -> np.ndarray:
        """Extract temperature data from HDF5 file."""
        return f["entry1/instrument_1/labview_data/LS_LLHTA"][()]

    def _display_roi_finder(self, f: h5py.File):
        """Display image for ROI selection."""
        patterns = np.squeeze(f["entry1/instrument_1/detector_1/data"][0, 0, :, :])
        plt.figure(figsize=(10, 8))
        plt.imshow(
            patterns,
            vmin=np.percentile(patterns, 5),
            vmax=np.percentile(patterns, 90),
        )
        plt.colorbar()
        plt.title("Select ROI from this image")
        plt.show()

    def _apply_normalization(self):
        """Apply normalization to the patterns."""
        if self.norm == "mean":
            patterns_mean = np.nanmean(self.patterns, axis=(1, 2))
            self.patterns = self.patterns / patterns_mean[:, None, None]
        elif self.norm == "sum":
            patterns_sum = np.sum(self.patterns, axis=(1, 2))
            self.patterns = self.patterns / patterns_sum[:, None, None]
        elif self.norm == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.norm}")

    # ========================================================================
    # ROI MANAGEMENT METHODS
    # ========================================================================

    def add_roi(self, roi_dict: Dict[str, Tuple]):
        """Add ROI(s) to the collection."""
        self.rois.update(roi_dict)

    def update_roi(self, roi_dict: Dict[str, Tuple]):
        """Update existing ROI(s)."""
        self.rois.update(roi_dict)

    def set_rois(self, roi_dict: Dict[str, Tuple]):
        """Replace all ROIs with new ones."""
        self.rois = roi_dict

    def get_roi_patterns(self, roi_name: str) -> np.ndarray:
        """Get patterns for a specific ROI."""
        if roi_name not in self.rois:
            raise KeyError(f"ROI '{roi_name}' not found")

        roi = self.rois[roi_name]
        y0, y1 = roi[0]
        x0, x1 = roi[1]
        return self.patterns[:, y0:y1, x0:x1]

    # ========================================================================
    # ANALYSIS METHODS
    # ========================================================================

    def compute_g2_for_roi(self, roi_name: str, device: str = None) -> np.ndarray:
        """Compute g2 correlation for a specific ROI."""
        if device is None:
            device = "cuda" if t.cuda.is_available() else "cpu"

        patterns_roi = self.get_roi_patterns(roi_name)
        return g2_fft_t(patterns_roi, device=device)

    def compute_g2_for_all_rois(self, device: str = None) -> Dict[str, np.ndarray]:
        """Compute g2 correlation for all ROIs."""
        results = {}
        for roi_name in self.rois:
            results[roi_name] = self.compute_g2_for_roi(roi_name, device)
        return results

    def compute_twotime_for_roi(self, roi_name: str, device: str = None) -> np.ndarray:
        """Compute 2-time correlation for a specific ROI."""
        if device is None:
            device = "cuda" if t.cuda.is_available() else "cpu"

        patterns_roi = self.get_roi_patterns(roi_name)

        twotime_correlation = twotime(patterns_roi, device=device)

        return twotime_correlation

    def compute_twotime_for_all_rois(self, device: str = None) -> Dict[str, np.ndarray]:
        """Compute 2-time correlation for all ROIs."""
        results = {}
        for roi_name in self.rois:
            results[roi_name] = self.compute_twotime_for_roi(roi_name, device)
        return results


# ============================================================================
# VISUALIZATION CLASS
# ============================================================================


class AndorDataVisualizer:
    """Class for visualizing Andor data and analysis results."""

    def __init__(self, data_loader: AndorDataLoader):
        self.data = data_loader

    def plot_sum_frames(self, log: bool = True):
        """Plot the sum of all frames."""
        plt.figure(figsize=(10, 8))
        plt.title(f"{self.data.filename_truncated} ({'log' if log else 'linear'})")

        if log:
            plt.imshow(np.log(self.data.patterns_sum))
        else:
            plt.imshow(self.data.patterns_sum)

        self._add_roi_overlays()
        plt.colorbar()
        plt.show()

    def plot_complete_overview(self):
        """Create a comprehensive 2x2 subplot overview."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(self.data.filename_truncated)

        # Plot mean patterns (log scale)
        self._plot_mean_patterns(axes[0, 0])

        # Plot g2 correlation functions
        self._plot_g2_correlations(axes[0, 1])

        # Plot temperature stability
        self._plot_temperature_stability(axes[1, 0])

        # Plot intensity stability
        self._plot_intensity_stability(axes[1, 1])

        plt.tight_layout()
        plt.show()

    def _plot_mean_patterns(self, ax):
        """Plot mean patterns with ROI overlays."""
        ax.imshow(np.log(self.data.patterns_mean))
        ax.set_title("Log Mean Patterns")
        self._add_roi_overlays(ax)

    def _plot_g2_correlations(self, ax):
        """Plot g2 correlation functions for all ROIs."""
        ax.set_title("G2 Correlation Functions")

        if self.data.rois:
            device = "cuda" if t.cuda.is_available() else "cpu"
            for roi_name, roi in self.data.rois.items():
                g2_result = self.data.compute_g2_for_roi(roi_name, device)
                ax.plot(
                    np.arange(1, g2_result.shape[0]) * self.data.periods,
                    np.nanmean(g2_result, axis=(1, 2))[1:],
                    label=roi_name,
                    color=roi_name,
                    alpha=0.7,
                )
                # TODO save ROIs g2 results to h5 file

        ax.set_xlabel("Lag Time [s]")
        ax.set_ylabel("G2 Correlation")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True)

    def _plot_temperature_stability(self, ax):
        """Plot temperature stability over time."""
        ax.plot(self.data.temps)
        ax.set_title("Temperature Stability")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Temperature")
        ax.grid(True)

    def _plot_intensity_stability(self, ax):
        """Plot intensity stability over time."""
        ax.plot(self.data.original_mean_intensity, label="Mean Intensity", color="blue")
        ax.set_title("Intensity Stability")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Mean Intensity")
        ax.grid(True)

    def _add_roi_overlays(self, ax=None):
        """Add ROI overlays to the current or specified axes."""
        if ax is None:
            ax = plt.gca()

        for roi_name, roi in self.data.rois.items():
            y0, y1 = roi[0]
            x0, x1 = roi[1]
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2,
                edgecolor=roi_name,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                roi_name,
                color="white",
                fontsize=8,
                ha="center",
                va="center",
            )


# ============================================================================
# CLASS for visualization of saved g2 results
# ============================================================================


class G2Visualizer:
    """
    Visualize g2 correlation results for different regions of interest (ROIs) from
    different saved data files.
    """

    def __init__(self, g2_data: Dict[str, Dict[str, np.ndarray]]):
        """
        Initialize the G2Visualizer with g2 data.

        Parameters
        ----------
        g2_data : dict
            Dictionary containing g2 correlation results for different ROIs.
            Format: {filename: {roi_name: g2_result}}
        """
        self.g2_data = g2_data

    def plot_g2_results(self):
        """Plot g2 correlation results for all ROIs across all files."""
        plt.figure(figsize=(12, 8))
        plt.title("G2 Correlation Results Across Files")

        for filename, rois in self.g2_data.items():
            for roi_name, g2_result in rois.items():
                plt.plot(
                    np.arange(1, g2_result.shape[0]),
                    np.nanmean(g2_result, axis=(1, 2))[1:],
                    label=f"{filename} - {roi_name}",
                    alpha=0.7,
                )

        plt.xlabel("Lag Time")
        plt.ylabel("G2 Correlation")
        plt.xscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()
