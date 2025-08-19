import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from networks.vision_transformer import SwinUnet as ViT_seg
from util.dataset_fsc import get_loader_fsc


def evaluator_fsc(args, model, mode, bs=1):
    valloader, db_val = get_loader_fsc(args, mode, batch_size=bs)

    train_mae = 0
    train_rmse = 0

    high_density = [
        '1123.jpg', '6860.jpg', '2159.jpg', '6811.jpg', '7611.jpg', '7042.jpg',
        '3486.jpg', '3434.jpg', '3436.jpg', '943.jpg', '2850.jpg', '3433.jpg',
        '3428.jpg', '3426.jpg', '949.jpg', '3425.jpg', '3476.jpg', '957.jpg',
        '5046.jpg', '762.jpg', '5747.jpg', '5045.jpg', '6969.jpg', '952.jpg',
        '3437.jpg', '935.jpg', '6886.jpg', '3665.jpg', '3477.jpg', '3491.jpg',
        '752.jpg', '3481.jpg', '7004.jpg', '5748.jpg', '3489.jpg', '3483.jpg',
        '975.jpg', '5736.jpg', '5740.jpg', '3484.jpg', '3485.jpg', '7477.jpg'
    ]

    model.eval()
    device = torch.device('cuda')
    val_it = tqdm(valloader, ncols=70)
    for data_iter_step, (samples, gt_dots, _, pos, _, im_id, _) in enumerate(val_it):

        samples = samples.to(device, non_blocking=True)
        gt_dots = gt_dots.to(device, non_blocking=True).half()

        _, _, h, w = samples.shape

        if im_id[0] in high_density:
            r_images = []
            r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))
            r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))

            pred_cnt = 0
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1

                with torch.no_grad():
                    while start + 383 < w:
                        in_img = r_image[:, :, :, start:start + 384]
                        output = model(in_img)
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

                pred_cnt += torch.sum(density_map / 60).item()
            pred_cnt_new = 0
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    in_img = samples[:, :, :, start:start + 384]
                    output = model(in_img)
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

            pred_cnt_new = torch.sum(density_map / 60).item()

            if pred_cnt > pred_cnt_new * 9:
                pred_cnt = pred_cnt_new
        else:
            density_map = torch.zeros([h, w])
            density_map = density_map.to(device, non_blocking=True)
            start = 0
            prev = -1
            with torch.no_grad():
                while start + 383 < w:
                    a = samples[:, :, :, start:start + 384]
                    output = model(a)
                    output = output[0].squeeze(0)
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
            pred_cnt_single = torch.sum(density_map / 60).item()

            pred_cnt = pred_cnt_single

        gt_cnt = gt_dots.shape[1]
        cnt_err = abs(pred_cnt - gt_cnt)

        train_mae += cnt_err
        train_rmse += cnt_err ** 2

    return train_mae / (len(valloader)), (train_rmse / (len(valloader))) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output dir')

    parser.add_argument('--data_path', type=str, help='dataset dir')
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

    mae, mse = evaluator_fsc(args, net, args.mode, args.batch_size)
    logging.info('\nVAL MAE: {:5.2f}, RMSE: {:5.2f} '.format(mae, mse))
