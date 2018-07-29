import cv2
import torch

import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
from data import VOCDetection, VOCroot, AnnotationTransform
from ssd import build_ssd
from data import VOC_CLASSES as labels

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def demo(img_id=0):
    net = build_ssd('test', 300, 21)  # initialize SSD
    print(net)
    net.load_weights('/media/sunwl/Datum/Projects/GraduationProject/SSD_VHR_300/weights/ssd300_mAP_77.43_v2.pth')
    testset = VOCDetection(VOCroot, [('2012', 'val')], None, AnnotationTransform())
    # image = testset.pull_image(img_id)
    image = cv2.imread('demos/02.png')
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)

    x = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
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
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    plt.show()


if __name__ == '__main__':
    demo(96)
