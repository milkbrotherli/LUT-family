"""Self-contained network-to-compact chroma-LUT conversion entry point."""

import argparse
import json
import os

import numpy as np
import torch

import model as Model
from checkpoint import load_checkpoint_state_dict
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


BLOCK_TO_STAGE = (
    ('convblock1', 1),
    ('convblock2', 2),
    ('downblock', 3),
    ('convblock3', 7),
    ('convblock4', 8),
    ('convblock5', 9),
)


def _expand_fine_coordinates(length, compressed_dimensions,
                             coordinate_arrays):
    if compressed_dimensions == 'xy':
        x, y = coordinate_arrays
        count = x.size
        c = np.tile(np.repeat(np.arange(length, dtype=np.int64), length), count)
        d = np.tile(np.arange(length, dtype=np.int64), count * length)
        return (
            np.repeat(x, length * length),
            np.repeat(y, length * length),
            c,
            d,
        )
    if compressed_dimensions == 'xyz':
        x, y, z = coordinate_arrays
        return (
            np.repeat(x, length),
            np.repeat(y, length),
            np.repeat(z, length),
            np.tile(np.arange(length, dtype=np.int64), x.size),
        )
    return coordinate_arrays


def _coarse_coordinates(length, coarse_step):
    base = np.arange(0, length, coarse_step, dtype=np.int64)
    mesh = np.meshgrid(base, base, base, base, indexing='ij')
    return tuple(axis.reshape(-1) for axis in mesh)


def _spatial_input(coordinates, interval):
    indices = np.stack(coordinates, axis=1)
    values = np.minimum(indices * (2 ** interval), 255).astype(np.float32)
    values /= 255.0
    return torch.from_numpy(values.reshape((-1, 1, 2, 2)))


def _channel_input(length, interval):
    base = np.arange(length, dtype=np.int64)
    mesh = np.meshgrid(base, base, base, indexing='ij')
    indices = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    values = np.minimum(indices * (2 ** interval), 255).astype(np.float32)
    values /= 255.0
    return torch.from_numpy(values.reshape((-1, 3, 1, 1)))


def _mode_input(input_tensor, mode):
    if mode == 's':
        return input_tensor
    output = torch.zeros(
        (input_tensor.shape[0], 1, 3, 3),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    if mode == 'd':
        output[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        output[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        output[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        output[:, :, 2, 2] = input_tensor[:, :, 1, 1]
    elif mode == 'y':
        output[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        output[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        output[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        output[:, :, 2, 1] = input_tensor[:, :, 1, 1]
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
    return output


def _evaluate_module(module, input_tensor, mode, batch_size):
    outputs = []
    module.eval()
    with torch.no_grad():
        for start in range(0, input_tensor.size(0), batch_size):
            batch = input_tensor[start:start + batch_size].cuda()
            if mode is not None:
                batch = _mode_input(batch, mode)
            output = module(batch)
            output = torch.round(torch.tanh(output) * 127)
            outputs.append(output.cpu().numpy().astype(np.int8))
    return np.concatenate(outputs, axis=0).reshape(input_tensor.size(0), -1)


def transfer_network_to_compact(
        channel, checkpoint_path, output_dir, nf=64, scale=1,
        lut_name='weight', interval=4, compressed_dimensions='xyzt',
        diagonal_width=3, sampling_interval=5, modes='sdy',
        batch_size=10000, ps_error=True):
    """Query a trained cc_u_alf network only at compact LUT sample points."""
    channel = channel.lower()
    if channel not in ('u', 'v'):
        raise ValueError('channel must be u or v')
    if sampling_interval <= interval:
        raise ValueError('sampling_interval must be larger than interval')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    model = Model.LUT_ILF_Net_cc_u_alf(
        nf=nf,
        scale=scale,
        stage1_modes=modes,
        weight=False,
        ps_error=ps_error,
    ).cuda()
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print('Loaded {} network checkpoint: {}'.format(channel.upper(), checkpoint_path))

    length, ref2index, fine_coordinates = _build_fine_region(
        interval, compressed_dimensions, diagonal_width
    )
    fine_coordinates = _expand_fine_coordinates(
        length, compressed_dimensions, fine_coordinates
    )
    coarse_step = 2 ** (sampling_interval - interval)
    coarse_coordinates = _coarse_coordinates(length, coarse_step)
    fine_input = _spatial_input(fine_coordinates, interval)
    coarse_input = _spatial_input(coarse_coordinates, interval)

    os.makedirs(output_dir, exist_ok=True)
    index_name = 'ref2index_{}{}i{}.npy'.format(
        compressed_dimensions, diagonal_width, sampling_interval
    )
    np.save(os.path.join(output_dir, index_name), ref2index)

    file_count = 1
    compact_bytes = ref2index.nbytes
    for mode in modes:
        for block_name, stage in BLOCK_TO_STAGE:
            block = getattr(model, block_name)
            module = block.module_dict['DepthwiseBlock0_{}'.format(mode)]
            key = '{}_s{}c0_{}'.format(lut_name, stage, mode)
            compact1 = _evaluate_module(module, fine_input, mode, batch_size)
            compact2 = _evaluate_module(module, coarse_input, mode, batch_size)
            path1 = os.path.join(output_dir, key + '_compress1.npy')
            path2 = os.path.join(output_dir, key + '_compress2.npy')
            np.save(path1, compact1)
            np.save(path2, compact2)
            file_count += 2
            compact_bytes += compact1.nbytes + compact2.nbytes
            print('Saved {}, {}'.format(path1, path2))

        for component in range(3):
            module = model.upblock.module_dict[
                'DepthwiseBlock{}_{}'.format(component, mode)
            ]
            key = '{}_s6c{}_{}'.format(lut_name, component, mode)
            compact1 = _evaluate_module(module, fine_input, mode, batch_size)
            compact2 = _evaluate_module(module, coarse_input, mode, batch_size)
            path1 = os.path.join(output_dir, key + '_compress1.npy')
            path2 = os.path.join(output_dir, key + '_compress2.npy')
            np.save(path1, compact1)
            np.save(path2, compact2)
            file_count += 2
            compact_bytes += compact1.nbytes + compact2.nbytes
            print('Saved {}, {}'.format(path1, path2))

    channel_input = _channel_input(length, interval)
    channel_lut = _evaluate_module(
        model.ChannelConv, channel_input, None, batch_size
    )
    channel_path = os.path.join(
        output_dir, '{}_s5_channel.npy'.format(lut_name)
    )
    np.save(channel_path, channel_lut)
    file_count += 1
    compact_bytes += channel_lut.nbytes

    metadata = {
        'source_type': 'network',
        'channel': channel,
        'checkpoint': os.path.normpath(checkpoint_path),
        'nf': nf,
        'scale': scale,
        'lut_name': lut_name,
        'interval': interval,
        'compressed_dimensions': compressed_dimensions,
        'diagonal_width': diagonal_width,
        'sampling_interval': sampling_interval,
        'modes': modes,
        'npy_file_count': file_count,
        'compact_bytes': compact_bytes,
    }
    with open(os.path.join(output_dir, 'compact_config.json'), 'w') as file:
        json.dump(metadata, file, indent=2)
    print('{} network-to-compact conversion complete: {}'.format(
        channel.upper(), output_dir
    ))
    return metadata


def build_parser(channel):
    parser = argparse.ArgumentParser(
        description='Transfer a trained {} network directly to compact LUTs'
        .format(channel.upper())
    )
    parser.add_argument('--transfer', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--loadDir', default='../checkpoints/{}'.format(channel))
    parser.add_argument('--checkpointPath', default=None)
    parser.add_argument('--loadIter', type=int, default=100000)
    parser.add_argument(
        '--expDir', default='../luts/compact-direct/{}'.format(channel)
    )
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--lutName', default='weight')
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--cd', choices=('xy', 'xyz', 'xyzt'), default='xyzt')
    parser.add_argument('--dw', type=int, default=3)
    parser.add_argument('--si', type=int, default=5)
    parser.add_argument('--stage1_modes', default='sdy')
    parser.add_argument('--transferBatchSize', type=int, default=10000)
    return parser


def transfer_cli(channel):
    opt = build_parser(channel).parse_args()
    checkpoint_path = opt.checkpointPath
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            opt.loadDir, 'Model_{:06d}.pth'.format(opt.loadIter)
        )
    transfer_network_to_compact(
        channel=channel,
        checkpoint_path=checkpoint_path,
        output_dir=opt.expDir,
        nf=opt.nf,
        scale=opt.scale,
        lut_name=opt.lutName,
        interval=opt.interval,
        compressed_dimensions=opt.cd,
        diagonal_width=opt.dw,
        sampling_interval=opt.si,
        modes=opt.stage1_modes,
        batch_size=opt.transferBatchSize,
        ps_error=True,
    )


if __name__ == '__main__':
    transfer_cli('u')
