"""Evaluate the bundled Y-channel regular LUTs.

Use --testRegularLUT to load model-official/regular-lut-pretrain at iteration
46000 and run a short Class-D validation on the bundled VVC data.  The script
can be launched from either the repository root or the filter directory.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model as Model
from data import Benchmark
from common.option import LUTTestOptions
from common.utils import PSNR


def configure_logger(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger('lut-ilf-regular-y-test')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d : %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(
            os.path.join(exp_dir, 'regular-y-test.txt'), mode='a'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def select_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is unavailable. Use --device cpu to run '
            'the slower CPU implementation.'
        )
    return torch.device(requested)


def lut_path(lut_dir, stem, load_iter):
    if load_iter > 0:
        filename = '{}_{:06d}.npy'.format(stem, load_iter)
    else:
        filename = stem + '.npy'
    return Path(lut_dir) / filename


def validate_lut_assets(opt):
    """Fail early with a useful message for wrong iterations or intervals."""
    length = 2 ** (8 - opt.interval) + 1
    spatial_rows = length ** 4
    expected = {}

    for mode in opt.stage1_modes:
        expected['{}_s1c0_{}'.format(opt.lutName, mode)] = (spatial_rows, 2)
        expected['{}_s2c0_{}'.format(opt.lutName, mode)] = (spatial_rows, 2)
        expected['{}_s4c0_{}'.format(opt.lutName, mode)] = (spatial_rows, 1)
        for channel in range(3):
            expected[
                '{}_s6c{}_{}'.format(opt.lutName, channel, mode)
            ] = (spatial_rows, opt.scale * opt.scale)
    expected['{}_s5_channel'.format(opt.lutName)] = (length ** 3, 3)

    missing = []
    shape_errors = []
    for stem, expected_shape in expected.items():
        path = lut_path(opt.lutDir, stem, opt.loadIter)
        if not path.is_file():
            missing.append(str(path))
            continue
        actual_shape = np.load(str(path), mmap_mode='r').shape
        if actual_shape != expected_shape:
            shape_errors.append(
                '{}: expected {}, found {}'.format(
                    path.name, expected_shape, actual_shape
                )
            )

    if missing:
        raise FileNotFoundError(
            'Regular LUT files are missing. Check --lutDir and --loadIter. '
            'First missing file: {}'.format(missing[0])
        )
    if shape_errors:
        raise ValueError(
            'The LUT shape does not match --interval/--scale. {}'.format(
                shape_errors[0]
            )
        )


def build_model(opt, device):
    model_class = (
        Model.LUT_ILF_Regular_LUT_RFd1_Test
        if opt.loadIter > 0
        else Model.LUT_ILF_Regular_LUT_RFd1
    )
    kwargs = dict(
        lut_folder=opt.lutDir,
        stages=opt.stages,
        modes=list(opt.stage1_modes),
        lutName=opt.lutName,
        upscale=opt.scale,
        interval=opt.interval,
        conversion=True,
    )
    if opt.loadIter > 0:
        kwargs['loadIter'] = opt.loadIter
    model = model_class(**kwargs).to(device)
    model.requires_grad_(False)
    model.eval()
    return model


def evaluate(model, valid, opt, device, logger):
    filtered_per_sequence = {}
    reconstructed_per_sequence = {}
    dataset = 'VVC_AI'

    with torch.no_grad():
        for original_name, reconstructed_name in zip(
                valid.files_ori[dataset], valid.files_rec[dataset]):
            original_frames = valid.ims_ori[
                '{}_{}'.format(dataset, original_name)
            ]
            reconstructed_frames = valid.ims_rec[
                '{}_{}'.format(dataset, reconstructed_name)
            ]
            filtered_psnr = []
            reconstructed_psnr = []

            for frame in sorted(original_frames.keys()):
                target = original_frames[frame]['y'][:, :, 0]
                reconstructed = reconstructed_frames[frame]['y'][:, :, 0]
                input_tensor = torch.from_numpy(
                    reconstructed.astype(np.float32).copy()
                ).unsqueeze(0).unsqueeze(0).div_(255.0).to(device)

                prediction = model(input_tensor, 'valid')
                prediction = prediction.squeeze(0).squeeze(0).cpu().numpy()
                prediction = np.round(np.clip(prediction, 0, 255)).astype(
                    np.uint8
                )

                filtered_psnr.append(PSNR(prediction, target, opt.scale))
                reconstructed_psnr.append(
                    PSNR(reconstructed, target, opt.scale)
                )

            filtered_per_sequence[original_name] = float(
                np.mean(filtered_psnr)
            )
            reconstructed_per_sequence[original_name] = float(
                np.mean(reconstructed_psnr)
            )
            logger.info(
                '%s | Rec PSNR: %.6f | LUT PSNR: %.6f | Difference: %.6f',
                original_name,
                reconstructed_per_sequence[original_name],
                filtered_per_sequence[original_name],
                filtered_per_sequence[original_name]
                - reconstructed_per_sequence[original_name],
            )

    average_filtered = float(np.mean(list(filtered_per_sequence.values())))
    average_reconstructed = float(
        np.mean(list(reconstructed_per_sequence.values()))
    )
    difference = average_filtered - average_reconstructed
    logger.info(
        'Dataset %s | Rec PSNR: %.6f | AVG LUT PSNR: %.6f | Difference: %.6f',
        dataset, average_reconstructed, average_filtered, difference,
    )
    return average_reconstructed, average_filtered, difference


def main():
    opt = LUTTestOptions(expected_mode='regular').parse()
    opt.lutDir = str(Path(opt.lutDir).resolve())
    opt.valDir = str(Path(opt.valDir).resolve())
    opt.expDir = str(Path(opt.expDir).resolve())

    logger = configure_logger(opt.expDir)
    logger.info('Options: %s', vars(opt))
    device = select_device(opt.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    logger.info('Device: %s', device)

    validate_lut_assets(opt)
    model = build_model(opt, device)
    valid = Benchmark(
        opt.valDir,
        opt.validPerSize,
        opt.validSkipSize,
        opt.qualityScale,
        validFast=not opt.fullValidation,
        scale=opt.scale,
    )
    rec_psnr, lut_psnr, difference = evaluate(
        model, valid, opt, device, logger
    )
    print(
        'Y Regular LUT | Rec PSNR: {:.6f} | LUT PSNR: {:.6f} | '
        'Difference: {:.6f}'.format(rec_psnr, lut_psnr, difference)
    )


if __name__ == '__main__':
    main()
