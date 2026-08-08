"""Self-contained compact chroma-LUT fine-tuning entry point."""

import argparse
import logging
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import model as Model
from data_uv import Benchmark, Provider

import sys
sys.path.insert(0, '../')
from common.utils import PSNR, logger_info


torch.backends.cudnn.benchmark = True


def _forward_chroma(model, channel, image_y, image_u, image_v, phase):
    if channel == 'u':
        return model(image_y, image_u, image_v, phase)
    return model(image_y, image_v, image_u, phase)


def evaluate_compact_chroma(model, valid, opt, iteration, channel, logger):
    logger.info('Compact {} validation is starting'.format(channel.upper()))
    sequence_filtered = {}
    sequence_reconstructed = {}

    with torch.no_grad():
        model.eval()
        dataset = 'VVC_AI'
        files_ori = valid.files_ori[dataset]
        files_rec = valid.files_rec[dataset]

        for file_ori, file_rec in zip(files_ori, files_rec):
            key_ori = '{}_{}'.format(dataset, file_ori)
            key_rec = '{}_{}'.format(dataset, file_rec)
            labels = valid.ims_ori[key_ori]
            inputs = valid.ims_rec[key_rec]
            filtered_psnr = []
            reconstructed_psnr = []

            logger.info('Valid Pair: ori ({}), rec ({})'.format(
                key_ori, key_rec
            ))
            for frame in range(opt.validPerSize):
                if frame % opt.validSkipSize:
                    continue
                label = labels[frame][channel]
                input_y = inputs[frame]['y'].astype(np.float32) / 255.0
                input_u = inputs[frame]['u'].astype(np.float32) / 255.0
                input_v = inputs[frame]['v'].astype(np.float32) / 255.0

                image_y = torch.from_numpy(
                    np.transpose(input_y, (2, 0, 1))[None]
                ).cuda()
                image_u = torch.from_numpy(
                    np.transpose(input_u, (2, 0, 1))[None]
                ).cuda()
                image_v = torch.from_numpy(
                    np.transpose(input_v, (2, 0, 1))[None]
                ).cuda()

                prediction = _forward_chroma(
                    model, channel, image_y, image_u, image_v, 'valid'
                )
                prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
                prediction = np.round(np.clip(prediction, 0, 255)).astype(
                    np.uint8
                )
                target = label[:, :, 0]
                reconstructed = inputs[frame][channel][:, :, 0]
                filtered_psnr.append(PSNR(prediction, target, opt.scale))
                reconstructed_psnr.append(
                    PSNR(reconstructed, target, opt.scale)
                )

            sequence_filtered[file_ori] = float(np.mean(filtered_psnr))
            sequence_reconstructed[file_ori] = float(
                np.mean(reconstructed_psnr)
            )
            logger.info(
                'Iter {} | {} | Rec PSNR: {:.6f} Val PSNR: {:.6f}'
                .format(
                    iteration, file_ori, sequence_reconstructed[file_ori],
                    sequence_filtered[file_ori],
                )
            )

    average_filtered = float(np.mean(list(sequence_filtered.values())))
    average_reconstructed = float(
        np.mean(list(sequence_reconstructed.values()))
    )
    difference = average_filtered - average_reconstructed
    logger.info(
        'Iter {} | Dataset VVC_AI | {} Rec PSNR: {:.6f}  AVG Val PSNR: '
        '{:.6f}  Difference: {:.6f}'.format(
            iteration, channel.upper(), average_reconstructed,
            average_filtered, difference,
        )
    )
    return difference


def _add_common_arguments(parser, channel):
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--cd', choices=('xy', 'xyz', 'xyzt'), default='xyzt')
    parser.add_argument('--dw', type=int, default=3)
    parser.add_argument('--si', type=int, default=5)
    parser.add_argument('--stage1_modes', default='sdy')
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--lutName', default='weight')
    parser.add_argument('--qualityScale', type=int, default=37)
    parser.add_argument('--valDir', default='../data/milkbrotherli/Benchmark_VVC')
    parser.add_argument('--validPerSize', type=int, default=20)
    parser.add_argument('--validSkipSize', type=int, default=10)
    parser.add_argument('--fullValidation', action='store_true')
    parser.add_argument('--gpuNum', type=int, default=1)
    parser.set_defaults(channel=channel, conversion=True)


def build_finetune_parser(channel):
    parser = argparse.ArgumentParser(
        description='Fine-tune compact {} LUTs'.format(channel.upper())
    )
    parser.add_argument('--fintune', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lutDir', default='../luts/compact/{}'.format(channel))
    parser.add_argument(
        '--lutSaveDir', default='../luts/compact-finetuned/{}'.format(channel)
    )
    parser.add_argument(
        '--expDir', default='../runs/finetune-compact/{}'.format(channel)
    )
    parser.add_argument('--trainDir', default='../data/milkbrotherli/DIV2K_YUV420')
    parser.add_argument('--trainDir2', default='../data/milkbrotherli/BVI-DVC')
    parser.add_argument('--dataReadDir', default='../data/milkbrotherli/dataRead')
    parser.add_argument('--datasetNum', type=int, default=2)
    parser.add_argument('--batchSize', type=int, default=2)
    parser.add_argument('--cropSize', type=int, default=8)
    parser.add_argument('--workerNum', type=int, default=0)
    parser.add_argument('--lr0', type=float, default=1e-3)
    parser.add_argument('--lr1', type=float, default=1e-4)
    parser.add_argument('--weightDecay', type=float, default=0)
    parser.add_argument('--startIter', type=int, default=0)
    parser.add_argument('--maxIter', type=int, default=100000)
    parser.add_argument('--displayStep', type=int, default=100)
    parser.add_argument('--valStep', type=int, default=1000)
    parser.add_argument('--saveStep', type=int, default=1000)
    parser.add_argument('--valStartIter', type=int, default=0)
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--skipInitialValidation', action='store_true')
    _add_common_arguments(parser, channel)
    return parser


def build_test_parser(channel):
    parser = argparse.ArgumentParser(
        description='Test compact fine-tuned {} LUTs'.format(channel.upper())
    )
    parser.add_argument('--testlut', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument(
        '--lutDir', default='../luts/compact-finetuned/{}'.format(channel)
    )
    parser.add_argument('--expDir', default='../runs/test-compact/{}'.format(channel))
    parser.add_argument('--loadIter', type=int, default=0)
    _add_common_arguments(parser, channel)
    return parser


def _make_model(opt, lut_folder, load_iter=0):
    return Model.LUT_ILF_Net_cc_u_alf_Compact_LUT(
        lut_folder=lut_folder,
        stages=opt.stages,
        modes=list(opt.stage1_modes),
        lutName=opt.lutName,
        upscale=opt.scale,
        interval=opt.interval,
        compressed_dimensions=opt.cd,
        diagonal_width=opt.dw,
        sampling_interval=opt.si,
        conversion=opt.conversion,
        load_iter=load_iter,
    ).cuda()


def _copy_compact_metadata(source_dir, destination_dir, opt):
    index_name = 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)
    source_index = os.path.join(source_dir, index_name)
    if not os.path.isfile(source_index):
        raise FileNotFoundError('Compact index not found: {}'.format(source_index))
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_index, os.path.join(destination_dir, index_name))
    source_config = os.path.join(source_dir, 'compact_config.json')
    if os.path.isfile(source_config):
        shutil.copy2(
            source_config, os.path.join(destination_dir, 'compact_config.json')
        )


def _save_compact_checkpoint(model, optimizer, scheduler, opt, iteration):
    os.makedirs(opt.lutSaveDir, exist_ok=True)
    torch.save(
        optimizer.state_dict(),
        os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(iteration)),
    )
    torch.save(
        scheduler.state_dict(),
        os.path.join(opt.expDir, 'Schedule_{:06d}.pth'.format(iteration)),
    )
    parameter_model = model.module if isinstance(
        model, torch.nn.DataParallel
    ) else model
    for name, parameter in parameter_model.named_parameters():
        lut_weight = np.round(
            np.clip(parameter.detach().cpu().numpy(), -1, 1) * 127
        ).astype(np.int8)
        np.save(
            os.path.join(
                opt.lutSaveDir, '{}_{:06d}.npy'.format(name, iteration)
            ),
            lut_weight,
        )
        np.save(os.path.join(opt.lutSaveDir, name + '.npy'), lut_weight)


def finetune_cli(channel):
    opt = build_finetune_parser(channel).parse_args()
    opt.validFast = not opt.fullValidation
    opt.valoutDir = os.path.join(opt.expDir, 'val')
    os.makedirs(opt.expDir, exist_ok=True)
    os.makedirs(opt.valoutDir, exist_ok=True)

    logger_name = 'lut-compact-{}-alf-ft'.format(channel)
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.txt'))
    logger = logging.getLogger(logger_name)
    logger.info(vars(opt))

    if opt.startIter > 0:
        model_folder = opt.lutSaveDir
        load_iter = opt.startIter
    else:
        model_folder = opt.lutDir
        load_iter = 0
        _copy_compact_metadata(opt.lutDir, opt.lutSaveDir, opt)

    model = _make_model(opt, model_folder, load_iter=load_iter)
    if opt.gpuNum > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(opt.gpuNum))
        )

    optimizer = optim.Adam(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=opt.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.weightDecay,
        amsgrad=False,
    )
    if opt.lr1 < 0:
        lr_lambda = lambda step: (
            ((1 + math.cos(step * math.pi / opt.maxIter)) / 2) * 0.8 + 0.2
        )
    else:
        lr_end = opt.lr1 / opt.lr0
        lr_start = 1 - lr_end
        lr_lambda = lambda step: (
            ((1 + math.cos(step * math.pi / opt.maxIter)) / 2) * lr_start
            + lr_end
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if opt.startIter > 0:
        optimizer_path = os.path.join(
            opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter)
        )
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        scheduler_path = os.path.join(
            opt.expDir, 'Schedule_{:06d}.pth'.format(opt.startIter)
        )
        if opt.schedule and os.path.isfile(scheduler_path):
            scheduler.load_state_dict(
                torch.load(scheduler_path, map_location='cpu')
            )
        else:
            scheduler.last_epoch = opt.startIter
            resumed_lrs = [
                base_lr * function(opt.startIter)
                for base_lr, function in zip(
                    scheduler.base_lrs, scheduler.lr_lambdas
                )
            ]
            for group, lr in zip(optimizer.param_groups, resumed_lrs):
                group['lr'] = lr
            scheduler._last_lr = resumed_lrs

    train_iter = Provider(
        opt.batchSize, opt.workerNum, opt.scale, opt.datasetNum,
        opt.trainDir, opt.trainDir2, opt.cropSize, 'YUV',
        opt.qualityScale, opt.dataReadDir,
    )
    valid = Benchmark(
        opt.valDir, opt.validPerSize, opt.validSkipSize,
        opt.qualityScale, opt.validFast, scale=opt.scale,
    )

    evaluation_model = model.module if isinstance(
        model, torch.nn.DataParallel
    ) else model
    if not opt.skipInitialValidation:
        evaluate_compact_chroma(
            evaluation_model, valid, opt, opt.startIter, channel, logger
        )

    accumulated_loss = 0.0
    data_time = 0.0
    run_time = 0.0
    last_iteration = opt.startIter
    for iteration in range(opt.startIter + 1, opt.maxIter + 1):
        last_iteration = iteration
        model.train()
        start = time.time()
        image_y, label_y, image_u, label_u, image_v, label_v = train_iter.next()
        image_y = image_y.cuda()
        image_u = image_u.cuda()
        image_v = image_v.cuda()
        target = label_u.cuda() if channel == 'u' else label_v.cuda()
        data_time += time.time() - start

        start = time.time()
        optimizer.zero_grad()
        prediction = _forward_chroma(
            model, channel, image_y, image_u, image_v, 'train'
        )
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        run_time += time.time() - start
        accumulated_loss += loss.item()

        if iteration % opt.displayStep == 0:
            logger.info(
                'Iter:{:6d}, lr:{:.8f}, loss:{:.4e}, dT:{:.4f}, rT:{:.4f}'
                .format(
                    iteration, optimizer.param_groups[0]['lr'],
                    accumulated_loss / opt.displayStep,
                    data_time / opt.displayStep, run_time / opt.displayStep,
                )
            )
            accumulated_loss = data_time = run_time = 0.0

        if (iteration % opt.valStep == 0 and
                iteration > opt.valStartIter):
            evaluation_model = model.module if isinstance(
                model, torch.nn.DataParallel
            ) else model
            evaluate_compact_chroma(
                evaluation_model, valid, opt, iteration, channel, logger
            )

        if iteration % opt.saveStep == 0:
            _save_compact_checkpoint(
                model, optimizer, scheduler, opt, iteration
            )
            logger.info('Checkpoint saved {}'.format(iteration))

    if (last_iteration > opt.startIter and
            last_iteration % opt.saveStep != 0):
        _save_compact_checkpoint(
            model, optimizer, scheduler, opt, last_iteration
        )
        logger.info('Final checkpoint saved {}'.format(last_iteration))

    logger.info('Fine-tuned compact LUTs saved to {}'.format(opt.lutSaveDir))
    logger.info('Complete')


def test_cli(channel):
    opt = build_test_parser(channel).parse_args()
    opt.validFast = not opt.fullValidation
    os.makedirs(opt.expDir, exist_ok=True)
    opt.valoutDir = os.path.join(opt.expDir, 'test')
    os.makedirs(opt.valoutDir, exist_ok=True)

    logger_name = 'lut-compact-{}-test'.format(channel)
    logger_info(logger_name, os.path.join(opt.expDir, logger_name + '.txt'))
    logger = logging.getLogger(logger_name)
    logger.info(vars(opt))

    model = _make_model(opt, opt.lutDir, load_iter=opt.loadIter)
    valid = Benchmark(
        opt.valDir, opt.validPerSize, opt.validSkipSize,
        opt.qualityScale, opt.validFast, scale=opt.scale,
    )
    difference = evaluate_compact_chroma(
        model, valid, opt, opt.loadIter, channel, logger
    )
    print('{} compact LUT PSNR difference: {:.6f}'.format(
        channel.upper(), difference
    ))
    logger.info('Complete')


if __name__ == '__main__':
    finetune_cli('u')
