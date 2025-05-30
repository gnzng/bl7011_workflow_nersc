import numpy as np
import torch as t
import h5py
import matplotlib.pyplot as plt


# calculate g2 using the fft on pytorch:
def g2_fft_t(
    img_stack: np.ndarray, device: str = "cpu", cut_last_ratio: float = 0.5
) -> np.ndarray:
    """
    Compute the g2 correlation function using FFT.

    Parameters
    ----------
    img_stack : numpy.ndarray or torch.Tensor
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
        # logger.debug(f"Reshaped img_stack to {img_stack.shape}")
    # Move input to GPU if it's not already there
    if not isinstance(img_stack, t.Tensor):
        img_stack = t.tensor(img_stack, dtype=t.float32)
    img_stack = img_stack.to(device)
    # Padding for FFT
    img_stack_padded = t.cat([img_stack, t.zeros_like(img_stack)], dim=0)
    # FFT calculations
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


def mean_every_n_frames(patterns, n):
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
    torch.tensor or None
        A tensor with shape (frames // n, height, width) containing the averaged frames.
        Returns None if the number of frames is not divisible by n, but should not happen.
    """
    patterns_subset = patterns[: patterns.shape[0] // n * n]

    if len(patterns_subset.shape) == 3:
        frames, height, width = patterns_subset.shape
        if frames % n != 0:
            # logger.error(f"Number of frames ({frames}) is not divisible by n ({n}).)")
            return None
        return patterns_subset.reshape(frames // n, n, height, width).mean(dim=1)

    elif len(patterns_subset.shape) == 2:
        frames, width = patterns_subset.shape
        height = 1
        if frames % n != 0:
            # logger.error(f"Number of frames ({frames}) is not divisible by n ({n}).)")
            return None
        return patterns_subset.reshape(frames // n, n, width, height).mean(dim=1)


class load_data_andor:
    def __init__(self, filename: str, norm="mean", roi=None):
        self.filename = filename
        self.filename_trun = filename.split("/")[-1]
        self.rois = dict()
        with h5py.File(filename, "r") as f:
            if roi == "find":
                patterns = np.squeeze(
                    f["entry1"]["instrument_1"]["detector_1"]["data"][0, 0, :, :]
                )
                plt.imshow(
                    patterns,
                    vmin=np.percentile(patterns, 5),
                    vmax=np.percentile(patterns, 90),
                )
                plt.colorbar()
                plt.show()
                return None, None

            elif roi is not None:
                try:
                    patterns = np.squeeze(
                        f["entry1"]["instrument_1"]["detector_1"]["data"][
                            :, :, roi[0] : roi[1], roi[2] : roi[3]
                        ]
                    )
                except KeyError:
                    # logger.warning(f"Could not roi {roi}")
                    patterns = np.squeeze(
                        f["entry1"]["instrument_1"]["detector_1"]["data"][:, :, :, :]
                    )
            else:
                patterns = np.squeeze(
                    f["entry1"]["instrument_1"]["detector_1"]["data"][:, :, :, :]
                )
            patterns = patterns.astype(np.float32)
            self.patterns = patterns
            periods = f["entry1"]["instrument_1"]["detector_1"]["period"][()]
            count_time = f["entry1"]["instrument_1"]["detector_1"]["count_time"][()]
            periods += count_time
            # readout_time = f["entry1"]["instrument_1"]["detector_1"]["detector_readout_time"][()]
            temps = f["entry1"]["instrument_1"]["labview_data"]["LS_LLHTA"][()]
            self.temps = temps

            # normalize patterns to the mean per frame:
            if norm == "mean":
                patterns_mean = np.mean(patterns, axis=(1, 2))
                # logger.debug(f"patterns_mean shape: {patterns_mean.shape}")
                patterns = patterns / patterns_mean[:, None, None]
                # logger.debug(f"patterns shape: {patterns.shape}")
            elif norm == "sum":
                patterns_sum = np.sum(patterns, axis=(1, 2))
                # logger.debug(f"patterns_sum shape: {patterns_sum.shape}")
                patterns = patterns / patterns_sum[:, None, None]
                # logger.debug(f"patterns shape: {patterns.shape}")
            elif norm == "none":
                pass
            else:
                raise Warning(f"Unknown normalization method: {norm}")

        # return patterns, periods, temps

    def plot_sum_frames(self, log=True):
        plt.figure()
        plt.title(self.filename_trun)
        if log:
            plt.imshow(np.log(np.nansum(self.patterns, axis=0)))
        else:
            plt.imshow(np.nansum(self.patterns, axis=0))

        if self.rois:
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
                plt.gca().add_patch(rect)
                plt.text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    roi_name,
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                )
        return plt.show()

    def plot_complete_overview(self):
        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(self.filename_trun)
        # Plot 1 (top-left)
        axes[0, 0].imshow(np.log(np.nanmean(self.patterns, axis=0)))
        axes[0, 0].set_title("nanmean")
        if self.rois:
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
                axes[0, 0].add_patch(rect)
                axes[0, 0].text(
                    (x0 + x1) / 2,
                    (y0 + y1) / 2,
                    roi_name,
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                )

        # Plot 2 (top-right)
        axes[0, 1].plot(np.nanmean(self.patterns, axis=(1, 2)))
        axes[0, 1].set_title("stability over time")
        axes[0, 1].set_xlabel("index")
        axes[0, 1].set_ylabel("mean intensity")
        axes[0, 1].grid(True)

        # Plot 3 [temp right now, but can be anything]
        axes[1, 0].plot(self.temps)
        axes[1, 0].set_title("temp stability over time")
        axes[1, 0].set_xlabel("index")
        axes[1, 0].set_ylabel("mean intensity")
        axes[1, 0].grid(True)

        # Plot 4 (bottom-right) will show the g2 correlation function all rois in self.rois
        axes[1, 1].set_title("g2 correlation function")
        if self.rois:
            for roi_name, roi in self.rois.items():
                y0, y1 = roi[0]
                x0, x1 = roi[1]
                patterns_roi = self.patterns[:, y0:y1, x0:x1]
                g2_result = g2_fft_t(patterns_roi)
                axes[1, 1].plot(
                    np.arange(1, g2_result.shape[0]),
                    np.nanmean(g2_result, axis=(1, 2))[1:],
                    label=roi_name,
                    color=roi_name,
                    alpha=0.7,
                )
        else:
            # no rois defined
            pass
        axes[1, 1].set_xlabel("lag time")
        axes[1, 1].set_ylabel("g2 correlation")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        # Set x-axis to log scale for the g2 correlation function plot
        axes[1, 1].set_xscale("log")

        # Adjust spacing between subplots
        plt.tight_layout()

        # Show the plot
        return plt.show()

    def add_g2_roi(self, new_roi):
        self.rois.update(new_roi)

    def update_g2_roi(self, new_roi):
        self.rois.update(new_roi)

    def set_g2_roi(self, new_roi):
        self.rois = new_roi
