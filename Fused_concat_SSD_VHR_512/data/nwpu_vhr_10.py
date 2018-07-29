"""
This very-high-resolution (VHR) remote sensing image dataset was constructed by Dr. Gong Cheng et al.
 from Northwestern Polytechnical University (NWPU).
"""
import os
import re
import cv2
import torch

import numpy as np
import torch.utils.data as Data

VHR_CLASSES = ('airplane', 'ship', 'storage tank', 'baseball diamond',
               'tennis court', 'basketball court', 'ground track field',
               'harbor', 'bridge,', 'vehicle')

root_path = "/media/sunwl/Datum/Datasets/NWPU VHR-10 dataset/ground truth"


def gen_train_val_test(root_path):
    imgs = os.listdir(path=root_path)
    rand = list(range(650))
    np.random.shuffle(rand)
    with open(os.path.join(root_path, "train2.txt"), 'w') as f:
        for x in rand[:195]:
            f.write(imgs[x].split('.')[0])
            f.write("\n")
    with open(os.path.join(root_path, "val2.txt"), 'w') as f:
        for x in rand[195:325]:
            f.write(imgs[x].split('.')[0])
            f.write('\n')

    with open(os.path.join(root_path, 'test2.txt'), 'w') as f:
        for x in rand[325:]:
            f.write(imgs[x].split('.')[0])
            f.write('\n')


class AnnotationTransform_VHR:
    """Transforms a VHR annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VHR's 10 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(zip(VHR_CLASSES, range(len(VHR_CLASSES))))

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an list e.g.['(123,10),(334,567),1',...]
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for pt in target[:-1]:
            cur_pt = re.findall(r'\d+', pt)
            cur_pt = [int(i) for i in cur_pt]
            print(cur_pt)
            cur_pt[0], cur_pt[2] = cur_pt[0] / width, cur_pt[2] / width
            cur_pt[1], cur_pt[3] = cur_pt[1] / height, cur_pt[3] / height
            res.append(cur_pt)

        return res


class VHRDetection(Data.Dataset):
    def __init__(self, root, image_sets, transform=None,
                 target_transform=None, dataset_name='VHR_10'):
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self._annopath = os.path.join('%s', "ground truth", '%s.txt')
        self._imgpath = os.path.join('%s', "positive image set", '%s.jpg')
        self.ids = list()

        for name in image_sets:
            for line in open(os.path.join(self.root, 'ground truth', name + '.txt')):
                self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        im, gt, w, h = self.pull_item(index)

        # return im, gt, w, h
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = open(self._annopath % img_id, 'r').read().split(' \n')
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.asarray(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
           index (int): index of img to show
        Return:
           PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = open(self._annopath % img_id, 'r').read().split(' \n')
        gt = self.target_transform(anno, 1, 1)

        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index=index)).unsqueeze_(0)


def detection_collate_VHR(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    # with open(os.path.join(root_path, "430.txt"), 'r') as f:
    #     file = f.read()
    #     s = file.split(' \n')
        # print(len(s))
        # print(re.findall(r'\d+', s))
        # res = AnnotationTransform()(s, 100, 100)
        # print(res)
        # ids = []
        # for line in open(os.path.join(root_path,'train.txt')):
        #     ids.append((root_path, line.strip()))
        # print(ids)
    gen_train_val_test(root_path)
