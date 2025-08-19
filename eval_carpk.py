import argparse
import logging
import os
import random
import sys

import hub
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg


def evaluator_carpk(model):
    ds_test = hub.load("hub://activeloop/carpk-test")
    dataloader_test = ds_test.pytorch(num_workers=8, batch_size=1, shuffle=False)

    logging.info("The length of val set is: {}".format(len(ds_test)))

    # some parameters in training
    train_mae = 0
    train_rmse = 0

    model.eval()
    device = torch.device('cuda')
    val_it = tqdm(dataloader_test, ncols=70)
    for data_iter_step, data in enumerate(val_it):
        samples = (data['images'] / 255).to(device, non_blocking=True)
        labels = data['labels'].to(device, non_blocking=True)
        samples = samples.transpose(2, 3).transpose(1, 2)

        pos = []
        for i in range(2):
            box = data['boxes'][0][i]
            pos.append(box)

        _, _, h, w = samples.shape

        r_images = []
        r_images.append(samples[0])

        pred_cnt = 0
        h = 384
        w = 683
        for r_image in r_images:
            r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1

            with torch.no_grad():
                while start + 383 < w:
                    output, = model(r_image[:, :, :, start:start + 384])
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            conv = nn.Conv2d(1, 1, kernel_size=(16, 16), stride=16, bias=False)
            conv.weight.data = torch.ones([1, 1, 16, 16]).to(device, non_blocking=True)

            density_map = density_map.unsqueeze(0)
            density_map = density_map.unsqueeze(0)
            d_m = conv(density_map / 60)
            pred_cnt += torch.sum(d_m).item()
            for i in range(d_m.shape[2]):
                for j in range(d_m.shape[3]):
                    if d_m[0][0][i][j] > 1.224:
                        pred_cnt -= 1

        gt_cnt = labels.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

        torch.cuda.synchronize()

    return train_mae / (len(dataloader_test)), (train_rmse / (len(dataloader_test))) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output dir')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int,
                        default=384, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--mode', default='val', type=str, help='val or test')

    args = parser.parse_args()
    config = get_config(args)
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    net = ViT_seg(config).cuda()
    net.load_from(config)

    logging.basicConfig(filename=args.output_dir + f"/{args.mode}.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    mae, mse = evaluator_carpk(net)
    logging.info('\nVAL MAE: {:5.2f}, RMSE: {:5.2f} '.format(mae, mse))
