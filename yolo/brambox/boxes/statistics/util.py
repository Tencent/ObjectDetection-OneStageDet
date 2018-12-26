#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#
#   Common functions for this package
#

__all__ = ['iou', 'ioa', 'match_detections']


def iou(a, b):
    """ Compute the intersection over union between two boxes.
    The function returns the ``IUO``, which is defined as:

    :math:`IOU = \\frac { {intersection}(a, b) } { {union}(a, b) }`

    Args:
        a (brambox.boxes.box.Box): First bounding box
        b (brambox.boxes.box.Box): Second bounding box

    Returns:
        Number: Intersection over union
    """
    intersection_area = intersection(a, b)
    union_area = a.width * a.height + b.width * b.height - intersection_area

    return intersection_area / union_area


def ioa(a, b, denominator='b'):
    """ Compute the intersection over area between two boxes a and b.
    The function returns the ``IOA``, which is defined as:

    :math:`IOA = \\frac { {intersection}(a, b) } { {area}(denominator) }`

    Args:
        a (brambox.boxes.box.Box): First bounding box
        b (brambox.boxes.box.Box): Second bounding box
        denominator (string, optional): String indicating from which box to compute the area; Default **'b'**

    Returns:
        Number: Intersection over union

    Note:
        The `denominator` can be one of 4 different values.
        If the parameter is equal to **'a'** or **'b'**, the area of that box will be used as the denominator.
        If the parameter is equal to **'min'**, the smallest of both boxes will be used
        and if it is equal to **'max'**, the biggest box will be used.
    """
    if denominator == 'min':
        div = min(a.width * a.height, b.width * b.height)
    elif denominator == 'max':
        div = max(a.width * a.height, b.width * b.height)
    elif denominator == 'a':
        div = a.width * a.height
    else:
        div = b.width * b.height

    return intersection(a, b) / div


def match_detections(detection_results, ground_truth, overlap_threshold, overlap_fn=iou):
    """ Match detection results with gound truth and return true and false positive rates.
    This function will return a list of values as the true and false positive rates.
    These values represent the rates at increasing confidence thresholds.

    Args:
        detection_results (dict): Detection objects per image
        ground_truth (dict): Annotation objects per image
        overlap_threshold (Number): Minimum overlap threshold for true positive
        overlap_fn (function, optional): Overlap area calculation function; Default :func:`~brambox.boxes.iou`

    Returns:
        list: **[true_positives]**, **[false_positives]**, **num_annotations**
    """
    positives = []
    num_annotations = 0

    # Make copy to not alter the reference
    detection_results = detection_results.copy()

    # make sure len(detection_results) == len(ground_truth) by inserting empty detections lists
    for image_id, annotations in ground_truth.items():
        if image_id not in detection_results:
            detection_results[image_id] = []

    for image_id, detections in detection_results.items():
        # Split ignored annotations
        annotations = []
        ignored_annotations = []
        for annotation in ground_truth[image_id][:]:
            if annotation.ignore:
                ignored_annotations.append(annotation)
            else:
                annotations.append(annotation)
        num_annotations += len(annotations)

        # Match detections
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for detection in detections:
            matched_annotation = match_detection_to_annotations(detection, annotations, overlap_threshold, overlap_fn)
            if matched_annotation is not None:
                del annotations[matched_annotation]
                # tp found
                positives.append((detection.confidence, True))
            elif match_detection_to_annotations(detection, ignored_annotations, overlap_threshold, ioa) is None:
                # fp found
                positives.append((detection.confidence, False))

    # sort matches by confidence from high to low
    positives = sorted(positives, key=lambda d: d[0], reverse=True)

    tps = []
    fps = []
    tp_counter = 0
    fp_counter = 0

    # all matches in dataset
    for pos in positives:
        if pos[1]:
            tp_counter += 1
        else:
            fp_counter += 1
        tps.append(tp_counter)
        fps.append(fp_counter)

    return tps, fps, num_annotations


def intersection(a, b):
    """ Calculate the intersection area between two boxes.

    Args:
        a (brambox.boxes.box.Box): First bounding box
        b (brambox.boxes.box.Box): Second bounding box

    Returns:
        Number: Intersection area
    """
    intersection_top_left_x = max(a.x_top_left, b.x_top_left)
    intersection_top_left_y = max(a.y_top_left, b.y_top_left)
    intersection_bottom_right_x = min(a.x_top_left + a.width,  b.x_top_left + b.width)
    intersection_bottom_right_y = min(a.y_top_left + a.height, b.y_top_left + b.height)

    intersection_width = intersection_bottom_right_x - intersection_top_left_x
    intersection_height = intersection_bottom_right_y - intersection_top_left_y

    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    return intersection_width * intersection_height


def match_detection_to_annotations(detection, annotations, overlap_threshold, overlap_fn):
    """ Compute the best match (largest overlap area) between a given detection and a list of annotations.

    Args:
        detection (brambox.boxes.detections.Detection): Detection to match
        annotations (list): Annotations to search for the best match
        overlap_threshold (Number): Minimum overlap threshold to consider detection and annotation as matched
        overlap_fn (function): Overlap area calculation function
    """
    best_overlap = overlap_threshold
    best_annotation = None
    for i, annotation in enumerate(annotations):
        if annotation.class_label != detection.class_label:
            continue

        overlap = overlap_fn(annotation, detection)
        if overlap < best_overlap:
            continue
        best_overlap = overlap
        best_annotation = i

    return best_annotation
