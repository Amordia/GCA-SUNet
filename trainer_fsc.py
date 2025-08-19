import logging
import math
import os
import sys

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.lr_sched as lr_sched
from util.dataset_fsc import get_loader_fsc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from val_fsc import evaluator_fsc


def trainer_fsc(args, model, snapshot_path, config):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    batch_size = args.batch_size
    base_lr = args.lr

    trainloader, db_train = get_loader_fsc(args, mode='train', batch_size=batch_size)

    model.load_from(config)
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), eps=0.00001)

    loss_scaler = NativeScaler()
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    min_MAE = 99999
    mae = 10000000
    mse = 10000000

    mae_test = 0
    mse_test = 0

    start_epoch = args.start
    iterator = tqdm(range(max_epoch - start_epoch), ncols=70)
    for epoch_num in iterator:
        optimizer.zero_grad()
        train_mae = 0
        train_rmse = 0
        model.train(True)
        accum_iter = args.accum_iter
        batch_loss = 0.

        batch_it = tqdm(trainloader, ncols=70)

        for i_batch, sampled_batch in enumerate(batch_it):
            lr_sched.adjust_learning_rate(optimizer, i_batch / len(trainloader) + epoch_num, args)

            image_batch, gt_density, exps, gt_map = sampled_batch[0], sampled_batch[1], sampled_batch[2], \
                sampled_batch[
                    -1]
            image_batch, gt_density, exps, gt_map = image_batch.cuda(), gt_density.cuda(), exps.cuda(), gt_map.cuda()
            with torch.cuda.amp.autocast():
                outputs = model(image_batch.float())

            # Compute loss function
            mask = np.random.binomial(n=1, p=0.8, size=[384, 384])
            masks = np.tile(mask, (outputs.shape[0], 1))
            masks = masks.reshape(outputs.shape[0], 384, 384)
            masks = torch.from_numpy(masks).to(outputs.device)
            loss_l2 = (outputs - gt_density) ** 2
            loss_l2 = (loss_l2 * masks / (384 * 384)).sum() / outputs.shape[0]

            loss = loss_l2
            loss_value = loss.item()
            if loss_value < 10 == False:
                logging.info(loss_value.dtype)

            batch_loss += loss_value

            if not math.isfinite(loss_value):
                logging.info("\nLoss is {}, stopping training".format(loss_value))
                sys.exit(1)

            batch_mae = 0
            batch_rmse = 0
            pred_cnt_list = []
            gt_cnt_list = []
            output_list = []
            for i in range(outputs.shape[0]):
                pred_cnt = torch.sum(outputs[i] / 60).item()
                pred_cnt_list.append(pred_cnt)
                output_list.append(outputs[i])
                gt_cnt = torch.sum(gt_density[i] / 60).item()
                gt_cnt_list.append(gt_cnt)
                cnt_err = abs(pred_cnt - gt_cnt)
                batch_mae += cnt_err
                batch_rmse += cnt_err ** 2

            train_mae += batch_mae
            train_rmse += batch_rmse

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(i_batch + 1) % accum_iter == 0)
            if (i_batch + 1) % accum_iter == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()

            iter_num = iter_num + 1
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('info/lr', lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_l2', loss_l2, iter_num)
            writer.add_scalar('info/mae', batch_mae, iter_num)
            writer.add_scalar('info/mse', batch_rmse, iter_num)

        epoch_mae = train_mae / (len(db_train))
        epoch_rmse = (train_rmse / (len(db_train))) ** 0.5
        epoch_loss = batch_loss / len(db_train)
        logging.info(
            '\nCurrent Loss: {:5.2f}, MAE: {:5.2f}, RMSE: {:5.2f} '.format(epoch_loss, epoch_mae, epoch_rmse))
        writer.add_scalar('train/mae', epoch_mae, epoch_num + start_epoch)
        writer.add_scalar('train/mse', epoch_rmse, epoch_num + start_epoch)
        writer.add_scalar('train/loss', epoch_loss, epoch_num + start_epoch)

        if args.output_dir and ((epoch_num + start_epoch) % 50 == 0 or epoch_num + start_epoch + 1 == max_epoch):
            save_model(model, snapshot_path, epoch_num + start_epoch, optimizer, loss_scaler, args)
        if args.output_dir and epoch_mae < min_MAE:
            min_MAE = epoch_mae
            save_model(model, snapshot_path, 666, optimizer, loss_scaler, args)

        if args.output_dir and (epoch_num + start_epoch) >= 0 and (epoch_num + start_epoch) % 2 == 0:
            mae_new, mse_new = evaluator_fsc(args=args, model=model, mode='val')
            logging.info('\nVAL MAE: {:5.2f}, RMSE: {:5.2f} '.format(mae_new, mse_new))
            writer.add_scalar('val/mae', mae_new, epoch_num + start_epoch)
            writer.add_scalar('val/mse', mse_new, epoch_num + start_epoch)
            if mae_new < mae:
                mae = mae_new
                mse = mse_new
                save_model(model, snapshot_path, epoch_num + start_epoch + 1, optimizer, loss_scaler, args)

    logging.info("min MAE: {}".format(min_MAE))
    logging.info("Best VAL MAE: {}, Best VAL RMSE: {}, Best TEST MAE: {}, Best TEST RMSE: {}"
                 .format(mae, mse, mae_test, mse_test))
    writer.close()
    return "Training Finished!"


def save_model(model, snapshot_path, epoch_num, optimizer, loss_scaler, args):
    epoch_name = str(epoch_num)
    if loss_scaler is not None:
        checkpoint_path = os.path.join(snapshot_path, f'ckpt-{epoch_name}.pth')
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_num,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        torch.save(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch_num}
        model.save_checkpoint(save_dir=snapshot_path, tag="checkpoint-%s" % epoch_name, client_state=client_state)
    logging.info("save model to {}".format(snapshot_path))
