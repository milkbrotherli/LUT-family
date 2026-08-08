"""Fine-tune the bundled RF-1 Y Compact LUT."""

import json
import random
import shutil
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
from common.option import CompactLUTFineTuneOptions


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


def compact_index_name(opt):
    return 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)


def build_model(opt, device):
    model_class = (
        Model.LUT_ILF_Compact_LUT_RFd1_Test
        if opt.inputIter > 0
        else Model.LUT_ILF_Compact_LUT_RFd1
    )
    kwargs = dict(
        lut_folder=opt.inputLUTDir,
        stages=opt.stages,
        modes=list(opt.stage1_modes),
        lutName=opt.lutName,
        upscale=opt.scale,
        interval=opt.interval,
        compressed_dimensions=opt.cd,
        diagonal_width=opt.dw,
        sampling_interval=opt.si,
        conversion=True,
    )
    if opt.inputIter > 0:
        kwargs['loadIter'] = opt.inputIter
    return model_class(**kwargs).to(device)


def save_luts(model, opt, iteration):
    output_dir = Path(opt.lutSaveDir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for name, parameter in model.named_parameters():
        lut_weight = np.round(
            np.clip(parameter.detach().cpu().numpy(), -1, 1) * 127
        ).astype(np.int8)

        latest_path = output_dir / '{}.npy'.format(name)
        iteration_path = output_dir / '{}_iter{:05d}.npy'.format(
            name, iteration
        )
        np.save(str(latest_path), lut_weight)
        np.save(str(iteration_path), lut_weight)
        saved_files.extend((latest_path.name, iteration_path.name))

    # ref2index defines the Compact layout and is not trainable.
    index_name = compact_index_name(opt)
    source_index = Path(opt.inputLUTDir) / index_name
    if not source_index.is_file():
        raise FileNotFoundError(
            'Compact index file not found: {}'.format(source_index)
        )
    shutil.copy2(str(source_index), str(output_dir / index_name))
    saved_files.append(index_name)
    return saved_files


def finetune_compact_lut(opt):
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

        if iteration % opt.displayStep == 0 or iteration == 1:
            print(
                'Iter {:6d}/{:6d} | lr {:.6e} | MSE {:.8e}'.format(
                    iteration,
                    opt.maxIter,
                    optimizer.param_groups[0]['lr'],
                    final_loss,
                )
            )

        if iteration % opt.saveStep == 0 or iteration == opt.maxIter:
            saved_files = save_luts(model, opt, iteration)
            print('Saved fine-tuned Compact LUTs at iteration', iteration)

    record = {
        'task': 'Fine-tune pretrained Y RF-1 Compact LUT',
        'source_directory': project_relative(opt.inputLUTDir),
        'source_iteration': opt.inputIter,
        'output_directory': project_relative(output_dir),
        'model_class': 'LUT_ILF_Compact_LUT_RFd1_Test',
        'interval': opt.interval,
        'compressed_dimensions': opt.cd,
        'diagonal_width': opt.dw,
        'sampling_interval': opt.si,
        'modes': opt.stage1_modes,
        'stages': opt.stages,
        'scale': opt.scale,
        'quality_scale': opt.qualityScale,
        'dataset_count': opt.datasetNum,
        'batch_size': opt.batchSize,
        'crop_size': opt.cropSize,
        'worker_count': opt.workerNum,
        'learning_rate': opt.lr0,
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
    print('Y Compact-LUT fine-tuning complete:', output_dir)
    return record


if __name__ == '__main__':
    options = CompactLUTFineTuneOptions().parse()
    finetune_compact_lut(options)
