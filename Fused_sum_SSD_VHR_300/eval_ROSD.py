import os
import re
import sys
import time
import torch
import argparse
import pickle

import numpy as np
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from data import ROSDroot
from data import ROSD_CLASSES as labelmap
from data import AnnotationTransform_ROSD, ROSDDetection, BaseTransform, ROSD_CLASSES
from ssd import build_ssd


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model',
                    default='/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_300/weights/Fused_ssd300_vhr_40000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_300/eval/',
                    type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--ROSD_root', default=ROSDroot, help='Location of ROSD root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.ROSD_root, "ground truth", '%s.txt')
imgpath = os.path.join(args.ROSD_root, "positive image set", '%s.jpg')
imgsetpath = os.path.join(args.ROSD_root, "ground truth", '{:s}.txt')
dataset_mean = (104, 117, 123)
set_type = 'test'
root_path = '/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_300'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    target = open(filename, 'r').read().split(' \n')
    res = []
    for pt in target[:-1]:
        obj_struct = {}
        cur_pt = re.findall(r'\d+', pt)
        cur_pt = [int(i) for i in cur_pt]
        # print(cur_pt)
        obj_struct['name'] = ROSD_CLASSES[cur_pt[-1]-1]
        obj_struct['difficult'] = 0
        obj_struct['bbox'] = cur_pt[:-1]

        res.append(obj_struct)

    return res


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(root_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_vhr_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VHR results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(root_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imgsetpath.format(set_type), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 2010年以前按recall等间隔取11个不同点处的精度值做平均(0.,0.1,0.2,...,0.9,1.0)
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                # 取最大值等价于2010以后先计算包络线的操作，保证precise非减
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # 2010年以后取所有不同的recall对应的点处的精度值做平均
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # 计算包络线，从后往前取最大保证precise非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # 找出所有检测结果中recall不同的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, cachedir,
             ovthresh=0.5, use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    计算每个类别对应的AP，mAP是所有类别AP的平均值
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots 提取所有测试图片中当前类别所对应的所有ground_truth
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class 提取所有测试图片中当前类别所对应的所有ground_truth
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # 找出所有当前类别对应的object
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # 该图片中该类别对应的所有bbox
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        det = [False] * len(R)
        # 累计所有图片中的该类别目标的总数，不算diffcult
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    # 读取相应类别的检测结果文件，每一行对应一个检测目标
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        # 某一行对应的检测目标所属的图像名
        image_ids = [x[0] for x in splitlines]
        # 读取该目标对应的置信度
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        # 将该类别的检测结果按照置信度大小降序排列
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        # 该类别检测结果的总数（所有检测出的bbox的数目）
        nd = len(image_ids)
        # 用于标记每个检测结果是tp还是fp
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        # 按置信度遍历每个检测结果
        for d in range(nd):
            # 取出该条检测结果所属图片中的所有ground truth
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            # 计算与该图片中所有ground truth的最大重叠度
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            # 如果最大的重叠度大于一定的阈值
            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    # 如果对应的最大重叠度的ground truth以前没被匹配过则匹配成功，即tp
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    # 若之前有置信度更高的检测结果匹配过这个ground truth，则此次检测结果为fp
                    else:
                        fp[d] = 1.
            # 该图片中没有对应类别的目标ground truth或者与所有ground truth重叠度都小于阈值
            else:
                fp[d] = 1.

        # 按置信度取不同数量检测结果时的累计fp和tp
        # np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # 召回率为占所有真实目标数量的比例，非减的，注意npos本身就排除了difficult，因此npos=tp+fn
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        # 精度为取的所有检测结果中tp的比例
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # 计算recall-precise曲线下面积（严格来说并不是面积）
        ap = voc_ap(rec, prec, use_07_metric)
    # 如果这个类别对应的检测结果为空，那么都是-1
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=512, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(os.path.join(root_path, 'ssd300_VHR_90000'), set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]  # [200,5]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()  # .t()->转置 [200,5]
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_vhr_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    num_classes = len(ROSD_CLASSES) + 1  # +1 background
    net = build_ssd('test', 512, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    dataset = ROSDDetection(args.ROSD_root, ['test'], BaseTransform(512, dataset_mean),
                           AnnotationTransform_ROSD())
    print(dataset)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 512,
             thresh=args.confidence_threshold)

