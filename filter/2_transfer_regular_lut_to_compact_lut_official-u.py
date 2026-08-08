"""Self-contained regular-to-compact chroma-LUT conversion entry point."""

import argparse
import json
import os

import numpy as np


SPATIAL_STAGES = (1, 2, 3, 7, 8, 9)
UP_CHANNELS = (0, 1, 2)


def _regular_lut_path(folder, stem, load_iter):
    if load_iter > 0:
        path = os.path.join(folder, '{}_{:06d}.npy'.format(stem, load_iter))
    else:
        path = os.path.join(folder, stem + '.npy')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            'Fine-tuned regular LUT not found: {}. Use --loadIter 0 for '
            'iteration-free latest files.'.format(path)
        )
    return path


def _build_fine_region(interval, compressed_dimensions, diagonal_width):
    length = 2 ** (8 - interval) + 1
    diag = 2 * diagonal_width + 1

    if compressed_dimensions == 'xy':
        ref2index = np.full((length, diag), -1, dtype=np.int64)
        coordinates = []
        for x in range(length):
            for y in range(length):
                if abs(x - y) <= diagonal_width:
                    ref2index[x, y - x] = len(coordinates)
                    coordinates.append((x, y))
    elif compressed_dimensions == 'xyz':
        ref2index = np.full((length, diag, diag), -1, dtype=np.int64)
        coordinates = []
        for x in range(length):
            for y in range(length):
                for z in range(length):
                    if (abs(x - y) <= diagonal_width and
                            abs(x - z) <= diagonal_width):
                        ref2index[x, y - x, z - x] = len(coordinates)
                        coordinates.append((x, y, z))
    elif compressed_dimensions == 'xyzt':
        ref2index = np.full((length, diag, diag, diag), -1, dtype=np.int64)
        coordinates = []
        for x in range(length):
            for y in range(length):
                for z in range(length):
                    for t in range(length):
                        if (abs(x - y) <= diagonal_width and
                                abs(x - z) <= diagonal_width and
                                abs(x - t) <= diagonal_width):
                            ref2index[x, y - x, z - x, t - x] = len(coordinates)
                            coordinates.append((x, y, z, t))
    else:
        raise ValueError('compressed_dimensions must be xy, xyz, or xyzt')

    coordinate_arrays = tuple(
        np.asarray(values, dtype=np.int64)
        for values in zip(*coordinates)
    )
    return length, ref2index, coordinate_arrays


def _compress_regular_lut(regular_lut, length, compressed_dimensions,
                          coordinate_arrays, coarse_step):
    if regular_lut.shape[0] != length ** 4:
        raise ValueError(
            'Expected {} rows for a 4-D regular LUT, got {}'.format(
                length ** 4, regular_lut.shape[0]
            )
        )

    output_channels = int(np.prod(regular_lut.shape[1:]))
    regular_lut = regular_lut.reshape(
        (length, length, length, length, output_channels)
    )

    if compressed_dimensions == 'xy':
        x, y = coordinate_arrays
        compress1 = regular_lut[x, y, :, :, :]
    elif compressed_dimensions == 'xyz':
        x, y, z = coordinate_arrays
        compress1 = regular_lut[x, y, z, :, :]
    else:
        x, y, z, t = coordinate_arrays
        compress1 = regular_lut[x, y, z, t, :]

    compress2 = regular_lut[
        ::coarse_step, ::coarse_step, ::coarse_step, ::coarse_step, :
    ]
    return (
        compress1.reshape((-1, output_channels)),
        compress2.reshape((-1, output_channels)),
    )


def transfer_finetuned_regular_to_compact(
        channel, lut_dir, output_dir, lut_name='weight', load_iter=0,
        interval=4, compressed_dimensions='xyzt', diagonal_width=3,
        sampling_interval=5, modes='sdy'):
    """Compress a fine-tuned regular U or V LUT set.

    The regular stage-5 3-D channel LUT is copied without 4-D compression.
    All spatial tables are sampled directly from the fine-tuned int8 arrays;
    no network evaluation or additional clipping is performed here.
    """
    channel = channel.lower()
    if channel not in ('u', 'v'):
        raise ValueError('channel must be u or v')
    if sampling_interval <= interval:
        raise ValueError(
            'sampling_interval must be greater than interval: {} <= {}'
            .format(sampling_interval, interval)
        )
    if diagonal_width < 0:
        raise ValueError('diagonal_width must be non-negative')

    coarse_ratio = 2 ** (sampling_interval - interval)
    length, ref2index, coordinate_arrays = _build_fine_region(
        interval, compressed_dimensions, diagonal_width
    )
    os.makedirs(output_dir, exist_ok=True)

    index_name = 'ref2index_{}{}i{}.npy'.format(
        compressed_dimensions, diagonal_width, sampling_interval
    )
    np.save(os.path.join(output_dir, index_name), ref2index)

    table_keys = []
    for mode in modes:
        table_keys.extend(
            's{}c0_{}'.format(stage, mode) for stage in SPATIAL_STAGES
        )
        table_keys.extend(
            's6c{}_{}'.format(component, mode) for component in UP_CHANNELS
        )

    regular_bytes = 0
    compact_bytes = ref2index.nbytes
    for key in table_keys:
        source_stem = '{}_{}'.format(lut_name, key)
        source_path = _regular_lut_path(lut_dir, source_stem, load_iter)
        regular_lut = np.load(source_path)
        regular_bytes += regular_lut.nbytes
        compress1, compress2 = _compress_regular_lut(
            regular_lut, length, compressed_dimensions,
            coordinate_arrays, coarse_ratio,
        )

        path1 = os.path.join(
            output_dir, '{}_compress1.npy'.format(source_stem)
        )
        path2 = os.path.join(
            output_dir, '{}_compress2.npy'.format(source_stem)
        )
        np.save(path1, compress1)
        np.save(path2, compress2)
        compact_bytes += compress1.nbytes + compress2.nbytes
        print('{} -> {}, {}'.format(source_path, path1, path2))

    channel_stem = '{}_s5_channel'.format(lut_name)
    channel_path = _regular_lut_path(lut_dir, channel_stem, load_iter)
    channel_lut = np.load(channel_path)
    expected_rows = length ** 3
    if channel_lut.shape[0] != expected_rows or int(
            np.prod(channel_lut.shape[1:])) != 3:
        raise ValueError(
            'Stage-5 channel LUT must have shape ({}, 3), got {}'
            .format(expected_rows, channel_lut.shape)
        )
    channel_output = os.path.join(output_dir, channel_stem + '.npy')
    np.save(channel_output, channel_lut.reshape((-1, 3)))
    regular_bytes += channel_lut.nbytes
    compact_bytes += channel_lut.nbytes

    metadata = {
        'channel': channel,
        'source_dir': os.path.normpath(lut_dir),
        'source_iter': load_iter,
        'lut_name': lut_name,
        'interval': interval,
        'compressed_dimensions': compressed_dimensions,
        'diagonal_width': diagonal_width,
        'sampling_interval': sampling_interval,
        'modes': modes,
        'spatial_regular_luts': len(table_keys),
        'compact_spatial_luts': len(table_keys) * 2,
        'regular_bytes': regular_bytes,
        'compact_bytes': compact_bytes,
        'compression_ratio': (
            float(regular_bytes) / compact_bytes if compact_bytes else 0.0
        ),
    }
    with open(os.path.join(output_dir, 'compact_config.json'), 'w') as file:
        json.dump(metadata, file, indent=2)

    print(
        '{} compact conversion complete: {} spatial pairs + stage 5; '
        'compression ratio {:.3f}x; output {}'.format(
            channel.upper(), len(table_keys), metadata['compression_ratio'],
            output_dir,
        )
    )
    return metadata


def build_transfer_parser(channel):
    parser = argparse.ArgumentParser(
        description=(
            'Transfer fine-tuned regular {} LUTs to compact LUTs'.format(
                channel.upper()
            )
        )
    )
    parser.add_argument('--transfer', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lutDir', default='../luts/finetuned/{}'.format(channel))
    parser.add_argument('--expDir', default='../luts/compact/{}'.format(channel))
    parser.add_argument('--lutName', default='weight')
    parser.add_argument('--loadIter', type=int, default=0)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--cd', choices=('xy', 'xyz', 'xyzt'), default='xyzt')
    parser.add_argument('--dw', type=int, default=3)
    parser.add_argument('--si', type=int, default=5)
    parser.add_argument('--stage1_modes', default='sdy')
    return parser


def transfer_cli(channel):
    opt = build_transfer_parser(channel).parse_args()
    transfer_finetuned_regular_to_compact(
        channel=channel,
        lut_dir=opt.lutDir,
        output_dir=opt.expDir,
        lut_name=opt.lutName,
        load_iter=opt.loadIter,
        interval=opt.interval,
        compressed_dimensions=opt.cd,
        diagonal_width=opt.dw,
        sampling_interval=opt.si,
        modes=opt.stage1_modes,
    )


if __name__ == '__main__':
    transfer_cli('u')
