import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the filter directory
from common.option import TestOptions
import model as Model
from checkpoint import load_checkpoint_state_dict


def get_input_tensor(opt):
    """Enumerate the 17^4 spatial-LUT inputs for interval=4."""
    base = torch.arange(0, 257, 2 ** opt.interval)
    base[-1] -= 1
    length = base.size(0)

    first = base.cuda().unsqueeze(1).repeat(1, length).reshape(-1)
    second = base.cuda().repeat(length)
    one_by_two = torch.stack([first, second], 1)

    third = base.cuda().unsqueeze(1).repeat(1, length * length).reshape(-1)
    one_by_two = one_by_two.repeat(length, 1)
    one_by_three = torch.cat([third.unsqueeze(1), one_by_two], 1)

    fourth = base.cuda().unsqueeze(1).repeat(1, length ** 3).reshape(-1)
    one_by_three = one_by_three.repeat(length, 1)
    one_by_four = torch.cat([fourth.unsqueeze(1), one_by_three], 1)

    return one_by_four.reshape(-1, 1, 2, 2).float() / 255.0


def get_input_tensor_3d(opt):
    """Enumerate [downsampled Y, U, V] for the 3-D channel LUT."""
    base = torch.arange(0, 257, 2 ** opt.interval)
    base[-1] -= 1
    length = base.size(0)

    first = base.cuda().unsqueeze(1).repeat(1, length).reshape(-1)
    second = base.cuda().repeat(length)
    one_by_two = torch.stack([first, second], 1)

    third = base.cuda().unsqueeze(1).repeat(1, length * length).reshape(-1)
    one_by_two = one_by_two.repeat(length, 1)
    one_by_three = torch.cat([third.unsqueeze(1), one_by_two], 1)

    return one_by_three.reshape(-1, 3, 1, 1).float() / 255.0


def get_mode_input_tensor(input_tensor, mode):
    if mode == "s":
        return input_tensor

    mode_input = torch.zeros(
        (input_tensor.shape[0], input_tensor.shape[1], 3, 3),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    if mode == "d":
        mode_input[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        mode_input[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        mode_input[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        mode_input[:, :, 2, 2] = input_tensor[:, :, 1, 1]
    elif mode == "y":
        mode_input[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        mode_input[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        mode_input[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        mode_input[:, :, 2, 1] = input_tensor[:, :, 1, 1]
    else:
        raise ValueError("Mode {} not implemented.".format(mode))

    return mode_input


def save_regular_lut(input_tensor, lut_path, module):
    """Evaluate one four-input network block and save its int8 LUT."""
    batch_size = max(1, input_tensor.size(0) // 100)
    outputs = []

    module.eval()
    with torch.no_grad():
        for start in range(0, input_tensor.size(0), batch_size):
            batch_input = input_tensor[start:start + batch_size]
            batch_output = module(batch_input)
            batch_output = torch.round(torch.tanh(batch_output) * 127)
            outputs.append(batch_output.cpu().numpy().astype(np.int8))

    results = np.concatenate(outputs, axis=0).reshape(input_tensor.size(0), -1)
    np.save(lut_path, results)
    print("Resulting LUT size:", results.shape, "Saved to", lut_path)


def transfer_lut_ilf_net_u_to_regular_lut(opt):
    if opt.weight:
        raise ValueError(
            "weight=True learns mode-fusion alphas, but the regular UV LUT runtime "
            "does not load those alphas. Use weight=False or add alpha export/runtime support."
        )

    os.makedirs(opt.lutDir, exist_ok=True)
    modes = list(opt.stage1_modes)

    model_g = Model.LUT_ILF_Net_cc_u_alf(
        nf=opt.nf,
        scale=opt.scale,
        stage1_modes=opt.stage1_modes,
        weight=opt.weight,
        ps_error=opt.ps_error,
    ).cuda()

    checkpoint_path = os.path.join(
        opt.loadDir, "Model_{:06d}.pth".format(opt.loadIter)
    )
    checkpoint = load_checkpoint_state_dict(checkpoint_path, map_location='cpu')
    model_g.load_state_dict(checkpoint, strict=True)
    model_g.eval()
    print("Loaded LUT_ILF_Net_cc_u_alf checkpoint:", checkpoint_path)

    spatial_input = get_input_tensor(opt)

    # Keep stages 1/2/3/5/6 compatible with the existing base UV LUT layout.
    # The ALF-only U branch is stored in stages 7/8/9.
    block_to_stage = (
        ("convblock1", 1),
        ("convblock2", 2),
        ("downblock", 3),
        ("convblock3", 7),
        ("convblock4", 8),
        ("convblock5", 9),
    )

    for mode in modes:
        mode_input = get_mode_input_tensor(spatial_input, mode)

        for block_name, stage in block_to_stage:
            block = getattr(model_g, block_name)
            module = block.module_dict["DepthwiseBlock0_{}".format(mode)]
            lut_path = os.path.join(
                opt.lutDir,
                "{}_s{}c0_{}.npy".format(opt.lutName, stage, mode),
            )
            save_regular_lut(mode_input, lut_path, module)

        # Stage 6 consumes the three outputs of ChannelConv independently.
        for channel in range(3):
            module = model_g.upblock.module_dict[
                "DepthwiseBlock{}_{}".format(channel, mode)
            ]
            lut_path = os.path.join(
                opt.lutDir,
                "{}_s6c{}_{}.npy".format(opt.lutName, channel, mode),
            )
            save_regular_lut(mode_input, lut_path, module)

    channel_input = get_input_tensor_3d(opt)
    channel_lut_path = os.path.join(
        opt.lutDir, "{}_s5_channel.npy".format(opt.lutName)
    )
    save_regular_lut(channel_input, channel_lut_path, model_g.ChannelConv)

    print("U LUT conversion complete:", opt.lutDir)


if __name__ == "__main__":
    options = TestOptions().parse()
    if os.path.normpath(options.loadDir) == os.path.normpath("../checkpoints"):
        options.loadDir = "../checkpoints/u"
    if os.path.normpath(options.lutDir) == os.path.normpath("../luts/regular"):
        options.lutDir = "../luts/regular/u"
    transfer_lut_ilf_net_u_to_regular_lut(options)
