#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#
#   Functions for generating miss-rate vs FPPI curves (False Positives Per Image) axis
#   and calculating log average miss-rate
#
import numpy as np
import scipy.interpolate

from .util import *

__all__ = ['mr_fppi', 'lamr']


def mr_fppi(detections, ground_truth, overlap_threshold=0.5):
    """ Compute a list of miss-rate FPPI values that can be plotted into a graph.

    Args:
        detections (dict): Detection objects per image
        ground_truth (dict): Annotation objects per image
        overlap_threshold (Number, optional): Minimum iou threshold for true positive; Default **0.5**

    Returns:
        tuple: **[miss-rate_values]**, **[fppi_values]**
    """
    num_images = len(ground_truth)
    tps, fps, num_annotations = match_detections(detections, ground_truth, overlap_threshold)

    miss_rate = []
    fppi = []
    for tp, fp in zip(tps, fps):
        miss_rate.append(1 - (tp / num_annotations))
        fppi.append(fp / num_images)

    return miss_rate, fppi


# TODO ? maarten -> why 9
def lamr(miss_rate, fppi, num_of_samples=9):
    """ Compute the log average miss-rate from a given MR-FPPI curve.
    The log average miss-rate is defined as the average of a number of evenly spaced log miss-rate samples
    on the :math:`{log}(FPPI)` axis within the range :math:`[10^{-2}, 10^{0}]`

    Args:
        miss_rate (list): miss-rate values
        fppi (list): FPPI values
        num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **9**

    Returns:
        Number: log average miss-rate
    """
    samples = np.logspace(-2., 0., num_of_samples)
    m = np.array(miss_rate)
    f = np.array(fppi)
    interpolated = scipy.interpolate.interp1d(f, m, fill_value=(1., 0.), bounds_error=False)(samples)
    log_interpolated = np.log(interpolated)
    avg = sum(log_interpolated) / len(log_interpolated)
    return np.exp(avg)
