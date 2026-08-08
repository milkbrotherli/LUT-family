import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model as Model
from checkpoint import find_model_checkpoint, load_checkpoint_state_dict


def get_input_tensor(opt):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)

    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)

    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(-1)
    onebythree = onebythree.repeat(L, 1)
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float() / 255.0

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

def compress_lut(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    input_tensor = input_tensor.reshape(L * L, L, L, 1, 2, 2)
    index_i = torch.zeros((N,)).type(torch.int64)
    index_j = torch.zeros((N,)).type(torch.int64)
    cnt = 0
    ref2index = np.zeros((L, diag), dtype=np.int_) - 1
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= d:
                index_i[cnt] = i
                index_j[cnt] = j
                ref2index[i, j - i] = cnt
                cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    index_compress = index_i * L + index_j
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut_xyz(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L, L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if abs(x - y) <= d and abs(x - z) <= d:
                    ref_x.append(x)
                    ref_y.append(y)
                    ref_z.append(z)
                    ref2index[x, y - x, z - x] = cnt
                    cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)

    index_compress = ref_x * L * L + ref_y * L + ref_z
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut_xyzt(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = opt.dw
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L * L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    ref_t = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    if abs(x - y) <= d and abs(x - z) <= d and abs(x - t) <= d:
                        ref_x.append(x)
                        ref_y.append(y)
                        ref_z.append(z)
                        ref_t.append(t)
                        ref2index[x, y - x, z - x, t - x] = cnt
                        cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)
    ref_t = torch.Tensor(ref_t).type(torch.int64)

    index_compress = ref_x * L * L * L + ref_y * L * L + ref_z * L + ref_t
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut_larger_interval(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    input_tensor = input_tensor.reshape(L, L, L, L, 1, 2, 2)

    if opt.si==5:
        k = 2
    elif opt.si==6:
        k = 4
    elif opt.si==7:
        k = 8
    else:
        raise ValueError

    compressed_input_tensor = input_tensor[::k, ::k, ::k, ::k, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def get_input_tensor_3D(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)

    # 3D input
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)

    input_tensor = onebythree.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 1, 3).float() / 255.0

    return input_tensor

def transfer_LUT_ILF_net_to_compact_lut(opt):
    def save_lut_ilf_compressed(x, lut_path, module):
        B = x.size(0) // 100
        outputs = []

        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]

                batch_output = module(batch_input)

                results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    # conversion
    modes = [i for i in opt.stage1_modes]
    stages = opt.stages

    model = getattr(Model, 'LUT_ILF_Net_RDd1')

    model_G = model(nf=opt.nf, scale=opt.scale, stage1_modes=opt.stage1_modes, stage2_modes=opt.stage2_modes, stage3_modes=opt.stage3_modes, stage4_modes=opt.stage4_modes, stages=stages, ps_error=opt.ps_error).cuda()

    checkpoint_path = find_model_checkpoint(
        opt.expDir, opt.loadIter, channel='Y'
    )
    lm = load_checkpoint_state_dict(checkpoint_path, map_location='cpu')
    model_G.load_state_dict(lm, strict=True)

    input_tensor_ori = get_input_tensor(opt)
    input_tensor_ori_3D = get_input_tensor_3D(opt)
    for mode in modes:
        input_tensor = input_tensor_ori.clone()
        if opt.cd == 'xyzt':
            input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
        elif opt.cd == 'xyz':
            input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
        elif opt.cd == 'xy':
            input_tensor_c1 = compress_lut(opt, input_tensor)
        else:
            raise ValueError
        input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

        if mode != 's':
            input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)
            input_tensor_c2 = get_mode_input_tensor(input_tensor_c2, mode)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, module)

        # conv2
        module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, module)

        # conv4
        module = model_G.convblock4.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, module)

        # conv6
        for c in range(3):
            module = model_G.upblock.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c1, lut_path, module)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c2, lut_path, module)

    # conv5
    input_tensor = input_tensor_ori_3D.reshape((-1, 3, 1, 1))
    module = model_G.ChannelConv
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_lut_ilf_compressed(input_tensor, lut_path, module)


if __name__ == "__main__":
    opt = TestOptions().parse()
    transfer_LUT_ILF_net_to_compact_lut(opt)
