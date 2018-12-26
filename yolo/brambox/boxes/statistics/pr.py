#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#
#   Functions for generating PR-curve values and calculating average precision
#

import math
from statistics import mean
import numpy as np
import scipy.interpolate

from .util import *

__all__ = ['pr', 'ap', 'voc_ap']


def pr(detections, ground_truth, overlap_threshold=0.5):
    """ Compute a list of precision recall values that can be plotted into a graph.

    Args:
        detections (dict): Detection objects per image
        ground_truth (dict): Annotation objects per image
        overlap_threshold (Number, optional): Minimum iou threshold for true positive; Default **0.5**

    Returns:
        tuple: **[precision_values]**, **[recall_values]**
    """
    tps, fps, num_annotations = match_detections(detections, ground_truth, overlap_threshold)

    precision = []
    recall = []
    for tp, fp in zip(tps, fps):
        recall.append(tp / num_annotations)
        precision.append(tp / (fp + tp))

    return precision, recall


def ap(precision, recall, num_of_samples=100):
    """ Compute the average precision from a given pr-curve.
    The average precision is defined as the area under the curve.

    Args:
        precision (list): Precision values
        recall (list): Recall values
        num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **100**

    Returns:
        Number: average precision
    """
    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0/num_of_samples)
        interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        # 1 point on PR: AP is box between (0,0) and (p,r)
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return avg


def voc_ap(prec, rec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    prec = np.array(prec)
    rec = np.array(rec)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
