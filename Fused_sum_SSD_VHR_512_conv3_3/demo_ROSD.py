import os
import cv2
import torch

import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from data import ROSDDetection, AnnotationTransform_ROSD, ROSDroot
from ssd import build_ssd
from data import ROSD_CLASSES as labels

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def demo_plt(img_id=0):
    net = build_ssd('test', 512, 5)  # initialize SSD
    print(net)
    net.load_weights('/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_512_conv3_3/weights/v2_rosd.pth')
    testset = ROSDDetection(ROSDroot, ['test'], None, AnnotationTransform_ROSD)
    image = testset.pull_image(img_id)
    # image = cv2.imread('demos/047.jpg')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)

    x = cv2.resize(rgb_image, (512, 512)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    #
    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 5)).tolist()
    plt.imshow(rgb_image.astype(np.uint8))  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            color = colors[i]
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    plt.savefig('/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_512_conv3_3/outputs/rosd_{:03}.png'.format(img_id))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def demo_cv2(img_id=0):
    net = build_ssd('test', 512, 5)  # initialize SSD
    print(net)
    net.load_weights('/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_512_conv3_3/weights/v2_rosd.pth')
    testset = ROSDDetection(ROSDroot, ['test'], None, AnnotationTransform_ROSD)
    image = testset.pull_image(img_id)
    # image = cv2.imread('demos/oiltank_1.jpg')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(rgb_image, (512, 512)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    colors = plt.cm.hsv(np.linspace(0, 1, 5)).tolist()
    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    im2show = np.copy(bgr_image)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            color = colors[i]
            color = [int(c * 255) for c in color[:3]]
            coords = pt[0], pt[1], pt[2], pt[3]
            cv2.rectangle(im2show, coords[0:2], coords[2:4], color, thickness=2)
            cv2.putText(im2show, display_txt, (int(coords[0]), int(coords[1]) - 3),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, color, thickness=1)
            j += 1
    cv2.imshow('original', bgr_image)
    cv2.imshow('demo', im2show)
    cv2.imwrite(os.path.join('/media/sunwl/Datum/Projects/GraduationProject/Fused_sum_SSD_VHR_512_conv3_3', "outputs",
                             "rosd_{:03d}.jpg".format(img_id)), im2show)
    cv2.waitKey(0)


if __name__ == '__main__':
    demo_plt(69)
    # demo_cv2(0)
    # for i in range(50,100):
    #     demo_cv2(i)

