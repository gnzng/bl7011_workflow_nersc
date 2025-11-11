import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure numpy warnings
np.seterr(divide="ignore")

# colors for rois based on matplotlib base colors
BASE_COLORS = ["red", "green", "blue", "cyan", "magenta", "yellow", "orange"]


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


class dataset_xpcs:
    """Class for loading and processing detector data from HDF5 files.
    Works with Andor and MTE detectors at COSMIC Scattering.
    """

    def __init__(self):
        """
        Initialize the data loader.
        """

    def load_file(
        self,
        filepath: str,
        filename: str,
        plot: bool = False,
        log_plot: bool = False,
        average_first_n: int = 1,
    ):
        """
        Parameters
            ----------
            filepath : str
                Path to the directory containing the data file
            filename : str
                Name of the data file or substring to search for
            plot : bool, optional
                Whether to display the first frame. Default is False.
            log_plot : bool, optional
                Whether to display the first frame with a logarithmic scale. Default is False.
            average_first_n : int, optional
                Number of initial frames to average. Default is 1 (no averaging).
        """
        self.filepath = Path(filepath)
        self.filename = filename

        # Find and validate file
        self.file_path = self._find_file()
        self.filename_truncated = self.file_path.name

        print(f"Found file: {self.file_path}")

        # Load data
        self.first_frame = self.load_first_frame(filepath, average_first_n)

        if plot:
            self.plot_single_frame_with_roi(log_plot=log_plot)

    def plot_single_frame_with_roi(self, log_plot: bool = False):
        """
        Plotting a single frame with optional ROIs overlaid.

        Parameters
        ----------
        rois : dict, optional
            Dictionary of ROIs to overlay on the image. Default is empty dict.
        log_plot : bool, optional
            Whether to use logarithmic scale for the image. Default is False.
        """

        if not hasattr(self, "first_frame"):
            raise AttributeError("First frame not loaded. Please load data first.")

        plt.figure(figsize=(10, 8))

        if log_plot:
            plt.imshow(
                np.log10(self.first_frame + 1),  # Add 1 to avoid log(0)
                vmin=np.percentile(np.log10(self.first_frame + 1), 5),
                vmax=np.percentile(np.log10(self.first_frame + 1), 90),
            )
        else:
            plt.imshow(
                self.first_frame,
                vmin=np.percentile(self.first_frame, 5),
                vmax=np.percentile(self.first_frame, 90),
            )
        if log_plot:
            plt.colorbar(label="Log10(Intensity + 1)")
        else:
            plt.colorbar(label="Intensity")

        if hasattr(self, "rois"):
            ax = plt.gca()
            for roi_name, roi in self.rois.items():
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
        plt.title(f"First frame of {self.filename_truncated}")
        plt.show()
        print(
            "If you want to do more plotting of this, you can access it via self.first_frame"
        )

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

    def load_first_frame(self, filepath: str, average_first_n: int) -> np.ndarray:
        """Load and return the first frame from the data file. And save other
        important metadata like periods and temperatures."""
        with h5py.File(self.file_path, "r") as f:
            if average_first_n > 1:
                n = int(average_first_n)
                patterns = np.squeeze(
                    f["entry1/instrument_1/detector_1/data"][0, :n, :, :]
                )
                patterns = np.mean(patterns, axis=0)
            else:
                patterns = np.squeeze(
                    f["entry1/instrument_1/detector_1/data"][0, 0, :, :]
                )
            self.periods = f["entry1/instrument_1/detector_1/period"][()]

            if b"Andor CCD" in f["entry1/instrument_1/detector_1/description"][()]:
                self.camera_type = "Andor CCD"
            # transform the periods for the MTE detector from millisconds to seconds
            elif b"MTE" in f["entry1/instrument_1/detector_1/description"][()]:
                self.periods *= 1e-3
                self.camera_type = "MTE3"
            else:
                self.camera_type = "Unknown"

            self.count_time = f["entry1/instrument_1/detector_1/count_time"][()]
            self.tempsA = f["entry1/instrument_1/labview_data/LS_LLHTA"][()]
            self.tempsB = f["entry1/instrument_1/labview_data/LS_LLHTB"][()]

        if patterns.ndim == 2:
            return patterns
        elif patterns.ndim == 3:
            return patterns[0, :, :]
        else:
            raise ValueError("Unexpected data shape for first frame.")

    def _load_data(self, roi: Optional[Union[str, Tuple]]):
        """Load data from HDF5 file."""
        with h5py.File(self.file_path, "r") as f:

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
                print(
                    "Temperature data not found in the file, initializing empty array."
                )
                self.temps = np.array([])

            # Store original data
            self.patterns = patterns.astype(np.float32)
            self.patterns_mean = np.nanmean(patterns, axis=0)
            # self.patterns_sum = np.nansum(patterns, axis=0)
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

        # check if the key is in BASE_COLORS if not warn the user
        for roi_name in roi_dict.keys():
            if roi_name not in BASE_COLORS:
                print(
                    f"Warning: ROI name '{roi_name}' is not a standard color name. Consider using one of {BASE_COLORS} for better visualization."
                )
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

        return self._load_roi_volume_patterns((y0, y1, x0, x1))

    def _load_roi_volume_patterns(self, roi: Tuple) -> np.ndarray:
        """Load volume patterns for a specific ROI."""
        with h5py.File(self.file_path, "r") as f:
            patterns = np.squeeze(
                f["entry1/instrument_1/detector_1/data"][
                    :, :, roi[0] : roi[1], roi[2] : roi[3]
                ]
            )
        return patterns.astype(np.float32)

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

        # this is a dictionary with g2 results for all rois
        self.g2_results_cubes = results

        # also compute the g2 curve (mean over all pixels) for all rois
        self.g2_results_curves = {}
        for roi_name, g2_cube in results.items():
            self.g2_results_curves[roi_name] = g2_cube.mean(axis=(1, 2))

    def plot_g2curve(self, roi_names: List[str] = None):
        """Plot g2 curve for a specific ROI."""
        if not hasattr(self, "g2_results_curves"):
            raise AttributeError(
                "g2 results not computed. Please run compute_g2_for_all_rois() first."
            )
        if roi_names is None:
            roi_names = list(self.g2_results_curves.keys())

        try:
            delay_times = self.periods * np.arange(
                len(next(iter(self.g2_results_curves.values())))
            )
        except Exception as e:
            print(
                f"Error computing delay times from periods: {e}. Using default frame indices instead."
            )
            delay_times = np.arange(len(next(iter(self.g2_results_curves.values()))))
            time_working = False

        plt.figure(figsize=(8, 5))
        for roi_name in roi_names:
            g2_curve = self.g2_results_curves[roi_name]
            if roi_name in BASE_COLORS:
                color = roi_name
            else:
                color = None

            time_working = True
            if color is not None:
                plt.plot(
                    delay_times[1:],
                    g2_curve[1:],
                    label=f"g2 Curve - ROI: {roi_name}",
                    color=color,
                )

            else:

                plt.plot(
                    delay_times[1:],
                    g2_curve[1:],
                    label=f"g2 Curve - ROI: {roi_name}",
                )
        if time_working:
            plt.xlabel("Time Delay (s)")
        else:
            plt.xlabel("Time Delay (frames)")
        plt.ylabel("g2")
        plt.title(f"g2 Curve for ROI: {roi_name}\n File: {self.filename_truncated}")
        plt.legend()
        plt.grid(True)
        plt.xscale("log")
        plt.show()
        print(
            "If you want to do more plotting of this, you can access it via self.g2_results_curves"
        )

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

        # this is a dictionary with 2-time results for all rois
        self.twotime_results = results

    def plot_twotime(self, roi_names: List[str] = None):
        """Plot 2-time correlation for a specific ROI."""
        if not hasattr(self, "twotime_results"):
            raise AttributeError(
                "2-time results not computed. Please run compute_twotime_for_all_rois() first."
            )

        if roi_names is None:
            roi_names = list(self.twotime_results.keys())

        for roi_name in roi_names:
            if roi_name not in self.twotime_results:
                raise KeyError(f"ROI '{roi_name}' not found in 2-time results.")

            twotime_matrix = self.twotime_results[roi_name]
            plt.figure(figsize=(8, 6))
            plt.imshow(
                twotime_matrix,
                origin="lower",
                aspect="auto",
                vmin=np.nanpercentile(twotime_matrix, 5),
                vmax=np.nanpercentile(twotime_matrix, 95),
                cmap="viridis",
            )
            plt.colorbar(label="2-Time Correlation")
            plt.xlabel("Time Frame")
            plt.ylabel("Time Frame")
            plt.title(
                f"2-Time Correlation for ROI: {roi_name}\n File: {self.filename_truncated}"
            )
            plt.show()
        print(
            "If you want to do more plotting of this, you can access it via self.twotime_results"
        )
