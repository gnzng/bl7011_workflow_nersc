import numpy as np
import torch as t

from BL7011 import twotime


# check if gpu is available:
print("CUDA available:", t.cuda.is_available())
filepath = "test_data/"


def test_twotime():
    # testing the twotime correlation function
    # on some example data as a numpy array

    brights = np.load(filepath + "220_twotime_brights.npy")

    # Calculate the two-time correlation
    g2 = twotime(brights, device="cuda" if t.cuda.is_available() else "cpu")

    # Check the shape of the result has to be (n_frames, n_frames)
    assert g2.shape == (brights.shape[0], brights.shape[0])
    # Check that the diagonal is NaN (self-correlation)
    assert t.isnan(g2.diagonal()).all()


if __name__ == "__main__":
    test_twotime()
