"""Fine-tune the bundled RF-1 Y Regular LUT."""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model as Model
from data import Provider
from common.option import LUTFineTuneOptions


def project_relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is unavailable. Use --device cpu for '
            'the slower CPU example.'
        )
    return torch.device(requested)


def build_model(opt, device):
    model_class = (
        Model.LUT_ILF_Regular_LUT_RFd1_Test
        if opt.inputIter > 0
        else Model.LUT_ILF_Regular_LUT_RFd1
    )
    kwargs = dict(
        lut_folder=opt.inputLUTDir,
        stages=opt.stages,
        modes=list(opt.stage1_modes),
        lutName=opt.lutName,
        upscale=opt.scale,
        interval=opt.interval,
        conversion=True,
    )
    if opt.inputIter > 0:
        kwargs['loadIter'] = opt.inputIter
    return model_class(**kwargs).to(device)


def save_luts(model, output_dir, iteration):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for name, parameter in model.named_parameters():
        lut_weight = np.round(
            np.clip(parameter.detach().cpu().numpy(), -1, 1) * 127
        ).astype(np.int8)

        latest_path = output_dir / '{}.npy'.format(name)
        iteration_path = output_dir / '{}_{:06d}.npy'.format(
            name, iteration
        )
        np.save(str(latest_path), lut_weight)
        np.save(str(iteration_path), lut_weight)
        saved_files.extend((latest_path.name, iteration_path.name))
    return saved_files


def finetune_regular_lut(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    device = select_device(opt.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    output_dir = Path(opt.lutSaveDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(opt, device)
    model.train()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=opt.weightDecay,
        amsgrad=False,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.maxIter,
        eta_min=opt.lr1,
    )

    provider = Provider(
        opt.batchSize,
        opt.workerNum,
        opt.scale,
        opt.datasetNum,
        opt.trainDir,
        opt.trainDir2,
        opt.cropSize,
        opt.colorSpace,
        opt.qualityScale,
        opt.dataReadDir,
    )
    # Provider historically moves batches to CUDA internally. Keep device
    # selection local so the same example can also run on CPU.
    provider.is_cuda = False

    final_loss = None
    saved_files = []
    start_time = time.time()
    for iteration in range(1, opt.maxIter + 1):
        reconstructed, target = provider.next()
        reconstructed = reconstructed.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        prediction = model(reconstructed, 'train')
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        current_lr = optimizer.param_groups[0]['lr']

        if iteration % opt.displayStep == 0 or iteration == 1:
            print(
                'Iter {:6d}/{:6d} | lr {:.6e} | MSE {:.8e}'.format(
                    iteration,
                    opt.maxIter,
                    current_lr,
                    final_loss,
                )
            )

        if iteration % opt.saveStep == 0 or iteration == opt.maxIter:
            saved_files = save_luts(model, output_dir, iteration)
            print('Saved fine-tuned LUTs at iteration', iteration)

        scheduler.step()

    record = {
        'task': 'Fine-tune pretrained Y RF-1 Regular LUT',
        'source_directory': project_relative(opt.inputLUTDir),
        'source_iteration': opt.inputIter,
        'output_directory': project_relative(output_dir),
        'model_class': 'LUT_ILF_Regular_LUT_RFd1_Test',
        'interval': opt.interval,
        'modes': opt.stage1_modes,
        'stages': opt.stages,
        'scale': opt.scale,
        'quality_scale': opt.qualityScale,
        'dataset_count': opt.datasetNum,
        'batch_size': opt.batchSize,
        'crop_size': opt.cropSize,
        'worker_count': opt.workerNum,
        'learning_rate': opt.lr0,
        'learning_rate_start': opt.lr0,
        'learning_rate_end': opt.lr1,
        'learning_rate_schedule': 'cosine',
        'weight_decay': opt.weightDecay,
        'iterations': opt.maxIter,
        'seed': opt.seed,
        'final_mse': final_loss,
        'elapsed_seconds': float(time.time() - start_time),
        'data_file_count': len(saved_files),
        'files': sorted(saved_files),
    }
    record_path = output_dir / opt.recordName
    with open(str(record_path), 'w', encoding='utf-8') as record_file:
        json.dump(record, record_file, indent=2)

    print('Fine-tuning record saved to:', record_path)
    print('Y Regular-LUT fine-tuning complete:', output_dir)
    return record


if __name__ == '__main__':
    options = LUTFineTuneOptions().parse()
    finetune_regular_lut(options)
