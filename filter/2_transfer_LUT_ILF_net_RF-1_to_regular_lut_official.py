import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.option import LUTTransferOptions
import model as Model
from checkpoint import load_checkpoint_state_dict


def project_relative(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def get_input_tensor(opt, device):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval, device=device)
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.repeat(L)
    onebytwo = torch.stack([first, second], 1)

    # 3D input
    third = base.unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)

    # 4D input
    fourth = base.unsqueeze(1).repeat(1, L * L * L).reshape(-1)
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float() / 255.0

    return input_tensor


def get_input_tensor_3D(opt, device):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval, device=device)
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.repeat(L)
    onebytwo = torch.stack([first, second], 1)

    # 3D input
    third = base.unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)

    input_tensor = onebythree.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 1, 3).float() / 255.0

    return input_tensor


def get_mode_input_tensor(input_tensor, mode):
    if mode == "d":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    elif mode == "y":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    return input_tensor


def transfer_LUT_ILF_net_to_regular_lut(opt):
    device_name = opt.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested but is unavailable. Use --device cpu for '
            'the slower CPU conversion.'
        )
    device = torch.device(device_name)
    os.makedirs(opt.lutDir, exist_ok=True)
    saved_files = []

    def save_LUT_ILF_regular_lut(x, lut_path, module):
        outputs = []

        # Extract input-output pairs
        with torch.no_grad():
            model_G.eval()
            for start in range(0, x.size(0), opt.transferBatchSize):
                batch_input = x[start:start + opt.transferBatchSize]
                batch_output = module(batch_input)
                results = torch.round(
                    torch.tanh(batch_output) * 127
                ).cpu().numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        saved_files.append(os.path.basename(lut_path))
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    # conversion
    modes = [i for i in opt.stage1_modes]
    stages = opt.stages

    model = Model.LUT_ILF_Net_RDd1

    model_G = model(
        nf=opt.nf,
        scale=opt.scale,
        stage1_modes=opt.stage1_modes,
        stage2_modes=opt.stage2_modes,
        stage3_modes=opt.stage3_modes,
        stage4_modes=opt.stage4_modes,
        stages=stages,
        ps_error=opt.ps_error,
    ).to(device)

    checkpoint_path = str(Path(opt.modelPath).resolve())
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'Y model checkpoint not found: {}'.format(checkpoint_path)
        )
    lm = load_checkpoint_state_dict(checkpoint_path, map_location='cpu')
    model_G.load_state_dict(lm, strict=True)
    print('Loaded Y model:', checkpoint_path)

    input_tensor_ori = get_input_tensor(opt, device)
    input_tensor_ori_3D = get_input_tensor_3D(opt, device)
    for mode in modes:
        input_tensor = input_tensor_ori.clone()
        if mode != 's':
            input_tensor = get_mode_input_tensor(input_tensor, mode)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.lutDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 1, mode))
        save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

        # conv2
        module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.lutDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 2, mode))
        save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

        # conv3
        # module = model_G.convblock3.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        # lut_path = os.path.join(opt.lutDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 3, mode))
        # save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

        # conv4
        module = model_G.convblock4.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.lutDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 4, mode))
        save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

        # conv6
        for c in range(3):
            module = model_G.upblock.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
            lut_path = os.path.join(opt.lutDir, '{}_s{}c{}_{}.npy'.format(opt.lutName, 6, c, mode))
            save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

    # conv5
    input_tensor = input_tensor_ori_3D.reshape((-1, 3, 1, 1))
    module = model_G.ChannelConv
    lut_path = os.path.join(opt.lutDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_LUT_ILF_regular_lut(input_tensor, lut_path, module)

    record = {
        'task': 'Y model to RF-1 Regular LUT',
        'source_model': project_relative(checkpoint_path),
        'source_iteration': opt.loadIter,
        'output_directory': project_relative(opt.lutDir),
        'model_class': 'LUT_ILF_Net_RDd1',
        'lut_runtime_class': 'LUT_ILF_Regular_LUT_RFd1',
        'nf': opt.nf,
        'interval': opt.interval,
        'modes': opt.stage1_modes,
        'stages': opt.stages,
        'scale': opt.scale,
        'lut_name': opt.lutName,
        'file_count': len(saved_files),
        'files': sorted(saved_files),
    }
    record_path = os.path.join(opt.lutDir, opt.recordName)
    with open(record_path, 'w', encoding='utf-8') as record_file:
        json.dump(record, record_file, indent=2)
    print('Transfer record saved to:', record_path)
    print('Y Regular-LUT conversion complete:', opt.lutDir)
    return record


if __name__ == "__main__":
    opt = LUTTransferOptions().parse()
    transfer_LUT_ILF_net_to_regular_lut(opt)
