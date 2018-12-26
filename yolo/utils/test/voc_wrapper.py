import pickle
import numpy as np
from PIL import Image
from .fast_rcnn.nms_wrapper import nms, soft_nms


def genResults(reorg_dets, results_folder, nms_thresh=0.45):
    for label, pieces in reorg_dets.items():
        ret = []
        dst_fp = '%s/comp4_det_test_%s.txt' % (results_folder, label)
        for name in pieces.keys():
            pred = np.array(pieces[name], dtype=np.float32)
            keep = nms(pred, nms_thresh, force_cpu=True)
            #keep = soft_nms(pred, sigma=0.5, Nt=0.3, method=1)
            #print k, len(keep), len(pred_dets[k])
            for ik in keep:
                #print k, pred_left[ik][-1], ' '.join([str(int(num)) for num in pred_left[ik][:4]])
                line ='%s %f %s' % (name, pred[ik][-1], ' '.join([str(num) for num in pred[ik][:4]]))
                ret.append(line)

        with open(dst_fp, 'w') as fd:
            fd.write('\n'.join(ret))


def reorgDetection(dets, netw, neth): #, prefix):
    reorg_dets = {}
    for k, v in dets.items():
        #img_fp = '%s/%s.jpg' % (prefix, k)
        img_fp = k #'%s/%s.jpg' % (prefix, k)
        #name = k.split('/')[-1]
        name = k.split('/')[-1][:-4]

        with Image.open(img_fp) as fd:
            orig_width, orig_height = fd.size
        scale = min(float(netw)/orig_width, float(neth)/orig_height)
        new_width = orig_width * scale
        new_height = orig_height * scale
        pad_w = (netw - new_width) / 2.0
        pad_h = (neth - new_height) / 2.0

        for iv in v:
            xmin = iv.x_top_left
            ymin = iv.y_top_left
            xmax = xmin + iv.width
            ymax = ymin + iv.height
            conf = iv.confidence
            class_label = iv.class_label
            #print(xmin, ymin, xmax, ymax)

            xmin = max(0, float(xmin - pad_w)/scale)
            xmax = min(orig_width - 1,float(xmax - pad_w)/scale)
            ymin = max(0, float(ymin - pad_h)/scale)
            ymax = min(orig_height - 1, float(ymax - pad_h)/scale)

            reorg_dets.setdefault(class_label, {})
            reorg_dets[class_label].setdefault(name, [])
            #line = '%s %f %f %f %f %f' % (name, conf, xmin, ymin, xmax, ymax)
            piece = (xmin, ymin, xmax, ymax, conf)
            reorg_dets[class_label][name].append(piece)

    return reorg_dets


def main():
    netw, neth = 416, 416
    results_folder = 'results_test'
    prefix = '/data2/yichaoxiong/data/VOCdevkit'
    with open('yolov2_bilinear_85000_416_bilinear.pkl', 'rb') as fd:
        dets = pickle.load(fd)
    reorg_dets = reorgDetection(dets, netw, neth, prefix)
    genResults(reorg_dets, results_folder)


if __name__ == '__main__':
    main()
