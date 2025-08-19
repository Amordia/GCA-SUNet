import json
import random

import numpy as np
import scipy.ndimage as ndimage
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from util.FSC147_384 import TransformTrain


# load data from FSC147
class TrainData(Dataset):
    def __init__(self, data_path, split='train'):
        anno_file = data_path + 'annotation_FSC147_384.json'
        data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
        im_dir = data_path + 'images_384_VarV2'
        gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

        with open(data_split_file) as f:
            data_split = json.load(f)
        with open(anno_file) as f:
            annotations = json.load(f)

        self.gt_dir = gt_dir
        self.im_dir = im_dir
        self.annotations = annotations
        self.img = data_split[split]
        random.shuffle(self.img)
        self.img_dir = im_dir

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        image.load()
        density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')
        m_flag = 1

        sample = {'image': image, 'lines_boxes': rects, 'gt_density': density, 'dots': dots, 'id': im_id,
                  'm_flag': m_flag, 'gt_map': 0}
        sample = TransformTrain(sample)
        return sample['image'], sample['gt_density'], sample['boxes'], sample['m_flag'], sample['gt_map']


class TestData(Dataset):
    def __init__(self, data_path, split='val'):
        anno_file = data_path + 'annotation_FSC147_384.json'
        data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
        im_dir = data_path + 'images_384_VarV2'
        gt_dir = data_path + 'gt_density_map_adaptive_384_VarV2'

        with open(data_split_file) as f:
            data_split = json.load(f)
        with open(anno_file) as f:
            annotations = json.load(f)

        self.gt_dir = gt_dir
        self.im_dir = im_dir
        self.annotations = annotations
        self.img = data_split[split]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        return self.get_img(im_id)

    def get_img(self, im_id):
        anno = self.annotations[im_id]
        bboxes = anno['box_examples_coordinates']

        dots = np.array(anno['points'])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        image.load()
        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor = float(new_W) / W
        image = transforms.Resize((new_H, new_W), antialias=True)(image)
        Normalize = transforms.Compose([transforms.ToTensor()])
        image = Normalize(image)

        rects = list()
        for bbox in bboxes:
            x1 = int(bbox[0][0] * scale_factor)
            y1 = bbox[0][1]
            x2 = int(bbox[2][0] * scale_factor)
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        boxes = list()
        scale_x = []
        scale_y = []
        cnt = 0
        for box in rects:
            cnt += 1
            if cnt > 3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            scale_x1 = torch.tensor((x2 - x1 + 1) / 384)
            scale_x.append(scale_x1)
            scale_y1 = torch.tensor((y2 - y1 + 1) / 384)
            scale_y.append(scale_y1)
            bbox = image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64), antialias=True)(bbox)
            boxes.append(bbox.numpy())
        scale_xx = torch.stack(scale_x).unsqueeze(-1)
        scale_yy = torch.stack(scale_y).unsqueeze(-1)
        scale = torch.cat((scale_xx, scale_yy), dim=1)
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        # Only for visualisation purpose, no need for ground truth density map indeed.
        gt_map = np.zeros((image.shape[1], image.shape[2]), dtype='float32')
        for i in range(dots.shape[0]):
            gt_map[min(new_H - 1, int(dots[i][1]))][min(new_W - 1, int(dots[i][0] * scale_factor))] = 1
        gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
        gt_map = torch.from_numpy(gt_map)
        gt_map = gt_map * 60

        sample = {'image': image, 'dots': dots, 'boxes': boxes, 'pos': rects, 'gt_map': gt_map, 'scale': scale}
        return sample['image'], sample['dots'], sample['boxes'], sample['pos'], sample['gt_map'], im_id, sample['scale']


def get_loader_fsc(args, mode='', batch_size=1):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    data_path = args.data_path

    if mode == 'train':
        dataset = TrainData(data_path, args.mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                worker_init_fn=worker_init_fn)
    else:
        dataset = TestData(data_path, mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                worker_init_fn=worker_init_fn)

    return dataloader, dataset
