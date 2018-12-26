from fast_rcnn.config import cfg, get_output_dir
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms, soft_nms
import cPickle
import os
import time

def im_detect(net, im, targe_size):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im = cv2.resize(im_orig, None, None, fx=float(targe_size)/float(im.shape[1]), fy=float(targe_size)/float(im.shape[0]), interpolation=cv2.INTER_LINEAR)
    blob = np.zeros((1, targe_size, targe_size, 3), dtype=np.float32)
    blob[0, :, :, :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    net.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
    net.blobs['data'].data[...] = blob
    detections = net.forward()['detection_out']

    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = np.minimum(np.maximum(detections[0, 0, :, 3] * im_orig.shape[1], 0), im_orig.shape[1] - 1)
    det_ymin = np.minimum(np.maximum(detections[0, 0, :, 4] * im_orig.shape[0], 0), im_orig.shape[0] - 1)
    det_xmax = np.minimum(np.maximum(detections[0, 0, :, 5] * im_orig.shape[1], 0), im_orig.shape[1] - 1)
    det_ymax = np.minimum(np.maximum(detections[0, 0, :, 6] * im_orig.shape[0], 0), im_orig.shape[0] - 1)
    dets = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label))

    keep_index = np.where(dets[:, 4] >= 0)[0]
    dets = dets[keep_index, :]
    return dets


def im_detect_ratio(net, im, targe_size1, targe_size2):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    if im_orig.shape[0] < im_orig.shape[1]:
        tmp = targe_size1
        targe_size1 = targe_size2
        targe_size2 = tmp
    im = cv2.resize(im_orig, None, None, fx=float(targe_size2)/float(im.shape[1]), fy=float(targe_size1)/float(im.shape[0]), interpolation=cv2.INTER_LINEAR)
    blob = np.zeros((1, int(targe_size1), int(targe_size2), 3), dtype=np.float32)
    blob[0, :, :, :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    net.blobs['data'].reshape(1, 3, im.shape[0], im.shape[1])
    net.blobs['data'].data[...] = blob
    detections = net.forward()['detection_out']

    det_label = detections[0, 0, :, 1]
    det_conf = detections[0, 0, :, 2]
    det_xmin = np.minimum(np.maximum(detections[0, 0, :, 3] * im_orig.shape[1], 0), im_orig.shape[1] - 1)
    det_ymin = np.minimum(np.maximum(detections[0, 0, :, 4] * im_orig.shape[0], 0), im_orig.shape[0] - 1)
    det_xmax = np.minimum(np.maximum(detections[0, 0, :, 5] * im_orig.shape[1], 0), im_orig.shape[1] - 1)
    det_ymax = np.minimum(np.maximum(detections[0, 0, :, 6] * im_orig.shape[0], 0), im_orig.shape[0] - 1)
    dets = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label))

    keep_index = np.where(dets[:, 4] >= 0)[0]
    dets = dets[keep_index, :]
    return dets


def flip_im_detect(net, im, targe_size):
    im_f = cv2.flip(im, 1)
    det_f = im_detect(net, im_f, targe_size)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t


def flip_im_detect_ratio(net, im, targe_size1, targe_size2):
    im_f = cv2.flip(im, 1)
    det_f = im_detect_ratio(net, im_f, targe_size1, targe_size2)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t


def vis_detections(im, class_name, dets, thresh=0.1):
    """Visual debugging of detections."""
    index = np.where(dets[:, -1] > thresh)[0]
    if index.shape[0] == 0:
        return
    det = dets[index,:]
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    plt.imshow(im)
    for i in xrange(det.shape[0]):
        bbox = det[i, :4]
        score = det[i, -1]
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='g', linewidth=3)
            )
        plt.text(bbox[0], bbox[1], '{}  {:.3f}'.format(class_name, score), color='r')
    plt.show()


def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets


def soft_bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= cfg.confidence_threshold)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


def single_scale_test_net(net, imdb, targe_size=(320, 320), vis=False):
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]
    output_dir = get_output_dir(imdb, net)

    net_time = 0
    for i in xrange(num_images):
        t = time.time()
        im = cv2.imread(imdb.image_path_at(i))
        det = im_detect_ratio(net, im, targe_size[0], targe_size[1])
        t2 = time.time() - t
        net_time += t2

        for j in xrange(1, imdb.num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if 'coco' in imdb.name:
                    keep = soft_nms(cls_dets, sigma=0.5, Nt=0.30, threshold=cfg.confidence_threshold, method=1)
                    cls_dets = cls_dets[keep, :]
                all_boxes[j][i] = cls_dets
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)

        print 'im_detect: {:d}/{:d} {:.4f}s'.format(i + 1, num_images, net_time / (i + 1))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if imdb.name == 'voc_2012_test':
        print 'Saving detections'
        imdb.config['use_salt'] = False
        imdb._write_voc_results_file(all_boxes)
    else:
        print 'Evaluating detections'
        imdb.evaluate_detections(all_boxes, output_dir)


def multi_scale_test_net_320(net, imdb, vis=False):
    targe_size = 320
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]
    output_dir = get_output_dir(imdb, net)

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        # ori and flip
        det0 = im_detect(net, im, targe_size)
        det0_f = flip_im_detect(net, im, targe_size)
        det0 = np.row_stack((det0, det0_f))

        det_r = im_detect_ratio(net, im, targe_size, int(0.6*targe_size))
        det_r_f = flip_im_detect_ratio(net, im, targe_size, int(0.6*targe_size))
        det_r = np.row_stack((det_r, det_r_f))

        # shrink: only detect big object
        det1 = im_detect(net, im, int(0.6*targe_size))
        det1_f = flip_im_detect(net, im, int(0.6*targe_size))
        det1 = np.row_stack((det1, det1_f))
        index = np.where(np.maximum(det1[:, 2] - det1[:, 0] + 1, det1[:, 3] - det1[:, 1] + 1) > 32)[0]
        det1 = det1[index, :]

        #enlarge: only detect small object
        det2 = im_detect(net, im, int(1.2*targe_size))
        det2_f = flip_im_detect(net, im, int(1.2*targe_size))
        det2 = np.row_stack((det2, det2_f))
        index = np.where(np.minimum(det2[:, 2] - det2[:, 0] + 1, det2[:, 3] - det2[:, 1] + 1) < 160)[0]
        det2 = det2[index, :]

        det3 = im_detect(net, im, int(1.4*targe_size))
        det3_f = flip_im_detect(net, im, int(1.4*targe_size))
        det3 = np.row_stack((det3, det3_f))
        index = np.where(np.minimum(det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 128)[0]
        det3 = det3[index, :]

        det4 = im_detect(net, im, int(1.6*targe_size))
        det4_f = flip_im_detect(net, im, int(1.6*targe_size))
        det4 = np.row_stack((det4, det4_f))
        index = np.where(np.minimum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1) < 96)[0]
        det4 = det4[index, :]

        det5 = im_detect(net, im, int(1.8*targe_size))
        det5_f = flip_im_detect(net, im, int(1.8*targe_size))
        det5 = np.row_stack((det5, det5_f))
        index = np.where(np.minimum(det5[:, 2] - det5[:, 0] + 1, det5[:, 3] - det5[:, 1] + 1) < 64)[0]
        det5 = det5[index, :]

        det7 = im_detect(net, im, int(2.2*targe_size))
        det7_f = flip_im_detect(net, im, int(2.2*targe_size))
        det7 = np.row_stack((det7, det7_f))
        index = np.where(np.minimum(det7[:, 2] - det7[:, 0] + 1, det7[:, 3] - det7[:, 1] + 1) < 32)[0]
        det7 = det7[index, :]

        # More scales make coco get better performance
        if 'coco' in imdb.name:
            det6 = im_detect(net, im, int(2.0*targe_size))
            det6_f = flip_im_detect(net, im, int(2.0*targe_size))
            det6 = np.row_stack((det6, det6_f))
            index = np.where(np.minimum(det6[:, 2] - det6[:, 0] + 1, det6[:, 3] - det6[:, 1] + 1) < 48)[0]
            det6 = det6[index, :]
            det = np.row_stack((det0, det_r, det1, det2, det3, det4, det5, det7, det6))
        else:
            det = np.row_stack((det0, det_r, det1, det2, det3, det4, det5, det7))

        for j in xrange(1, imdb.num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if 'coco' in imdb.name:
                    cls_dets = soft_bbox_vote(cls_dets)
                else:
                    cls_dets = bbox_vote(cls_dets)
                all_boxes[j][i] = cls_dets
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)

        print 'im_detect: {:d}/{:d}'.format(i + 1, num_images)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if imdb.name == 'voc_2012_test':
        print 'Saving detections'
        imdb.config['use_salt'] = False
        imdb._write_voc_results_file(all_boxes)
    else:
        print 'Evaluating detections'
        imdb.evaluate_detections(all_boxes, output_dir)


def multi_scale_test_net_512(net, imdb, vis=False):
    targe_size = 512
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]
    output_dir = get_output_dir(imdb, net)

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))

        # ori and flip
        det0 = im_detect(net, im, targe_size)
        det0_f = flip_im_detect(net, im, targe_size)
        det0 = np.row_stack((det0, det0_f))

        det_r = im_detect_ratio(net, im, targe_size, int(0.75*targe_size))
        det_r_f = flip_im_detect_ratio(net, im, targe_size, int(0.75*targe_size))
        det_r = np.row_stack((det_r, det_r_f))

        # shrink: only detect big object
        det_s1 = im_detect(net, im, int(0.5*targe_size))
        det_s1_f = flip_im_detect(net, im, int(0.5*targe_size))
        det_s1 = np.row_stack((det_s1, det_s1_f))

        det_s2 = im_detect(net, im, int(0.75*targe_size))
        det_s2_f = flip_im_detect(net, im, int(0.75*targe_size))
        det_s2 = np.row_stack((det_s2, det_s2_f))

        # #enlarge: only detect small object
        det3 = im_detect(net, im, int(1.75*targe_size))
        det3_f = flip_im_detect(net, im, int(1.75*targe_size))
        det3 = np.row_stack((det3, det3_f))
        index = np.where(np.minimum(det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 128)[0]
        det3 = det3[index, :]

        det4 = im_detect(net, im, int(1.5*targe_size))
        det4_f = flip_im_detect(net, im, int(1.5*targe_size))
        det4 = np.row_stack((det4, det4_f))
        index = np.where(np.minimum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1) < 192)[0]
        det4 = det4[index, :]

        # More scales make coco get better performance
        if 'coco' in imdb.name:
            det5 = im_detect(net, im, int(1.25*targe_size))
            det5_f = flip_im_detect(net, im, int(1.25*targe_size))
            det5 = np.row_stack((det5, det5_f))
            index = np.where(np.minimum(det5[:, 2] - det5[:, 0] + 1, det5[:, 3] - det5[:, 1] + 1) < 224)[0]
            det5 = det5[index, :]

            det6 = im_detect(net, im, int(2*targe_size))
            det6_f = flip_im_detect(net, im, int(2*targe_size))
            det6 = np.row_stack((det6, det6_f))
            index = np.where(np.minimum(det6[:, 2] - det6[:, 0] + 1, det6[:, 3] - det6[:, 1] + 1) < 96)[0]
            det6 = det6[index, :]

            det7 = im_detect(net, im, int(2.25*targe_size))
            det7_f = flip_im_detect(net, im, int(2.25*targe_size))
            det7 = np.row_stack((det7, det7_f))
            index = np.where(np.minimum(det7[:, 2] - det7[:, 0] + 1, det7[:, 3] - det7[:, 1] + 1) < 64)[0]
            det7 = det7[index, :]
            det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4, det5, det6, det7))
        else:
            det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4))

        for j in xrange(1, imdb.num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if 'coco' in imdb.name:
                    cls_dets = soft_bbox_vote(cls_dets)
                else:
                    cls_dets = bbox_vote(cls_dets)
                all_boxes[j][i] = cls_dets
                if vis:
                    vis_detections(im, imdb.classes[j], cls_dets)

        print 'im_detect: {:d}/{:d}'.format(i + 1, num_images)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if imdb.name == 'voc_2012_test':
        print 'Saving detections'
        imdb.config['use_salt'] = False
        imdb._write_voc_results_file(all_boxes)
    else:
        print 'Evaluating detections'
        imdb.evaluate_detections(all_boxes, output_dir)
