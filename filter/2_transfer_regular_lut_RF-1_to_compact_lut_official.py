"""Convert the bundled RF-1 Y Regular LUT into a Compact LUT."""

import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.option import LUTCompressionOptions


SPATIAL_STAGES = ((1, 2), (2, 2), (4, 1))
UP_CHANNELS = 3


def project_relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def regular_lut_path(folder, stem, load_iter):
    if load_iter > 0:
        filename = '{}_{:06d}.npy'.format(stem, load_iter)
    else:
        filename = stem + '.npy'
    return Path(folder) / filename


def compact_lut_path(folder, stem, load_iter):
    if load_iter > 0:
        filename = '{}_iter{:05d}.npy'.format(stem, load_iter)
    else:
        filename = stem + '.npy'
    return Path(folder) / filename


def build_fine_region_indices(length, compressed_dimensions, diagonal_width):
    diagonal_size = 2 * diagonal_width + 1

    if compressed_dimensions == 'xy':
        ref2index = np.full(
            (length, diagonal_size), -1, dtype=np.int32
        )
        groups = []
        for x in range(length):
            for y in range(length):
                if abs(x - y) <= diagonal_width:
                    ref2index[x, y - x] = len(groups)
                    groups.append(x * length + y)
        groups = np.asarray(groups, dtype=np.int64)
        trailing = np.arange(length ** 2, dtype=np.int64)
        flat_indices = (
            groups[:, None] * (length ** 2) + trailing[None, :]
        ).reshape(-1)

    elif compressed_dimensions == 'xyz':
        ref2index = np.full(
            (length, diagonal_size, diagonal_size), -1, dtype=np.int32
        )
        groups = []
        for x in range(length):
            for y in range(length):
                for z in range(length):
                    if (
                        abs(x - y) <= diagonal_width
                        and abs(x - z) <= diagonal_width
                    ):
                        ref2index[x, y - x, z - x] = len(groups)
                        groups.append((x * length + y) * length + z)
        groups = np.asarray(groups, dtype=np.int64)
        trailing = np.arange(length, dtype=np.int64)
        flat_indices = (
            groups[:, None] * length + trailing[None, :]
        ).reshape(-1)

    elif compressed_dimensions == 'xyzt':
        ref2index = np.full(
            (length, diagonal_size, diagonal_size, diagonal_size),
            -1,
            dtype=np.int32,
        )
        flat_indices = []
        compact_index = 0
        for x in range(length):
            for y in range(length):
                for z in range(length):
                    for t in range(length):
                        if (
                            abs(x - y) <= diagonal_width
                            and abs(x - z) <= diagonal_width
                            and abs(x - t) <= diagonal_width
                        ):
                            ref2index[x, y - x, z - x, t - x] = (
                                compact_index
                            )
                            flat_indices.append(
                                ((x * length + y) * length + z) * length + t
                            )
                            compact_index += 1
        flat_indices = np.asarray(flat_indices, dtype=np.int64)

    else:
        raise ValueError(
            'compressed dimensions must be xy, xyz, or xyzt'
        )

    return flat_indices, ref2index


def build_coarse_indices(interval, sampling_interval):
    if sampling_interval <= interval:
        raise ValueError('--si must be larger than --interval')
    length = 2 ** (8 - interval) + 1
    step = 2 ** (sampling_interval - interval)
    positions = np.arange(0, length, step, dtype=np.int64)
    if positions[-1] != length - 1:
        raise ValueError(
            'sampling interval does not retain the 255 endpoint'
        )
    x, y, z, t = np.meshgrid(
        positions, positions, positions, positions, indexing='ij'
    )
    return (
        ((x * length + y) * length + z) * length + t
    ).reshape(-1)


def load_regular_lut(opt, stem, expected_shape):
    path = regular_lut_path(opt.regularLUTDir, stem, opt.loadIter)
    if not path.is_file():
        raise FileNotFoundError(
            'Regular LUT file not found: {}'.format(path)
        )
    lut = np.load(str(path))
    if lut.shape != expected_shape:
        raise ValueError(
            '{} has shape {}, expected {}. Check --interval and the RF-1 '
            'structure.'.format(path.name, lut.shape, expected_shape)
        )
    return lut


def transfer_regular_lut_to_compact_lut(opt):
    output_dir = Path(opt.compactLUTDir)
    output_dir.mkdir(parents=True, exist_ok=True)

    length = 2 ** (8 - opt.interval) + 1
    regular_rows = length ** 4
    fine_indices, ref2index = build_fine_region_indices(
        length, opt.cd, opt.dw
    )
    coarse_indices = build_coarse_indices(opt.interval, opt.si)
    saved_files = []

    index_name = 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)
    np.save(str(output_dir / index_name), ref2index)
    saved_files.append(index_name)

    def compress_and_save(stem, output_channels):
        regular_lut = load_regular_lut(
            opt, stem, (regular_rows, output_channels)
        )
        for compact_part, indices in (
                (1, fine_indices), (2, coarse_indices)):
            output_stem = '{}_compress{}'.format(stem, compact_part)
            output_path = compact_lut_path(
                output_dir, output_stem, opt.loadIter
            )
            compact_lut = regular_lut[indices]
            np.save(str(output_path), compact_lut)
            saved_files.append(output_path.name)
            print(
                '{} -> {} {}'.format(
                    stem, output_path.name, compact_lut.shape
                )
            )

    for mode in opt.stage1_modes:
        for stage, output_channels in SPATIAL_STAGES:
            compress_and_save(
                '{}_s{}c0_{}'.format(opt.lutName, stage, mode),
                output_channels,
            )
        for channel in range(UP_CHANNELS):
            compress_and_save(
                '{}_s6c{}_{}'.format(opt.lutName, channel, mode),
                opt.scale * opt.scale,
            )

    # Stage 5 is a 3-D Channel LUT. It is retained without 4-D compression.
    channel_stem = '{}_s5_channel'.format(opt.lutName)
    channel_lut = load_regular_lut(
        opt, channel_stem, (length ** 3, 3)
    )
    channel_output = compact_lut_path(
        output_dir, channel_stem, opt.loadIter
    )
    np.save(str(channel_output), channel_lut)
    saved_files.append(channel_output.name)

    values_per_spatial_row = len(opt.stage1_modes) * (
        sum(output_channels for _, output_channels in SPATIAL_STAGES)
        + UP_CHANNELS * opt.scale * opt.scale
    )
    source_entries = (
        regular_rows * values_per_spatial_row + length ** 3 * 3
    )
    compact_entries = 0
    for filename in saved_files:
        if filename.startswith('weight_'):
            compact_entries += np.load(
                str(output_dir / filename), mmap_mode='r'
            ).size

    record = {
        'task': 'Y RF-1 Regular LUT to Compact LUT',
        'source_directory': project_relative(opt.regularLUTDir),
        'source_iteration': opt.loadIter,
        'output_directory': project_relative(output_dir),
        'regular_runtime_class': 'LUT_ILF_Regular_LUT_RFd1_Test',
        'compact_runtime_class': 'LUT_ILF_Compact_LUT_RFd1_Test',
        'interval': opt.interval,
        'compressed_dimensions': opt.cd,
        'diagonal_width': opt.dw,
        'sampling_interval': opt.si,
        'modes': opt.stage1_modes,
        'scale': opt.scale,
        'fine_region_rows': int(fine_indices.size),
        'coarse_region_rows': int(coarse_indices.size),
        'source_value_count': int(source_entries),
        'compact_value_count': int(compact_entries),
        'compression_ratio': float(source_entries / compact_entries),
        'data_file_count': len(saved_files),
        'files': sorted(saved_files),
    }
    record_path = output_dir / opt.recordName
    with open(str(record_path), 'w', encoding='utf-8') as record_file:
        json.dump(record, record_file, indent=2)

    print('Compression record saved to:', record_path)
    print('Y Compact-LUT conversion complete:', output_dir)
    return record


if __name__ == '__main__':
    options = LUTCompressionOptions().parse()
    transfer_regular_lut_to_compact_lut(options)
