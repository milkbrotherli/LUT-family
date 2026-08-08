import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model as Model


def get_input_tensor(opt):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    # base[-1] -= 1
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

    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2)

    return input_tensor

def get_input_tensor_3D(opt):
    base = torch.arange(0, 257, 2 ** opt.interval)
    # base[-1] -= 1
    L = base.size(0)

    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)

    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat([third.unsqueeze(1), onebytwo], 1)

    input_tensor = onebythree.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 1, 3)

    return input_tensor

def compress_lut(opt, input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-257, floor in LUT-LUT transfer
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
    np.save(os.path.join(opt.expDir, 'ref2index_{}{}i{}.npy'.format(opt.cd, opt.dw, opt.si)), ref2index)
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

def transfer_regular_lut_to_compact_lut(opt):
    def save_lut_ilf_compressed(x, lut_path, interval, weight):
        B = x.size(0) // 100
        weight = weight * 127
        weight = round_func(weight)
        weight = torch.clamp(weight, -127, 127)
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]
                batch_output = ConvertLUT(batch_input, interval, weight)
                results = batch_output.cpu().data.numpy().astype(np.int8)
                outputs += [results]
        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def ConvertLUT(batch, interval, weight):
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1
        floor_batch = torch.div(batch, q, rounding_mode='floor')
        floor_batch_a1 = floor_batch[:, :, 0, 0].flatten()
        floor_batch_b1 = floor_batch[:, :, 0, 1].flatten()
        floor_batch_c1 = floor_batch[:, :, 1, 0].flatten()
        floor_batch_d1 = floor_batch[:, :, 1, 1].flatten()
        output = weight[floor_batch_a1 * L * L * L + floor_batch_b1 * L * L + floor_batch_c1 * L + floor_batch_d1]

        return output

    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    # main
    modes = [i for i in opt.stage1_modes]
    stages = opt.stages
    model = Model.LUT_ILF_Regular_LUT_Test
    model_G = model(loadIter=opt.loadIter, lut_folder=opt.lutDir, stages=stages, modes=modes, lutName=opt.lutName, upscale=opt.scale, interval=opt.interval, conversion=opt.conversion).cuda()

    input_tensor = get_input_tensor(opt)
    for mode in modes:
        if opt.cd == 'xyzt':
            input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
        elif opt.cd == 'xyz':
            input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
        elif opt.cd == 'xy':
            input_tensor_c1 = compress_lut(opt, input_tensor)
        else:
            raise ValueError
        input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

        # conv1
        key = "s{}c0_{}".format(1, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv2
        key = "s{}c0_{}".format(2, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv3
        key = "s{}c0_{}".format(3, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 3, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 3, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv4
        key = "s{}c0_{}".format(4, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv6
        for c in range(4):
            key = "s{}c{}_{}".format(6, c, mode)
            key = "weight_" + key
            module = getattr(model_G, key)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

    # conv5
    key = "s{}_channel".format(5)
    key = "weight_" + key
    module = getattr(model_G, key)
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_lut_ilf_compressed(input_tensor, lut_path, opt.interval, module)

def transfer_regular_lut_to_compact_lut_together(opt, model):
    def save_lut_ilf_compressed(x, lut_path, interval, weight):
        B = x.size(0) // 100
        weight = weight * 127
        weight = round_func(weight)
        weight = torch.clamp(weight, -127, 127)
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]
                batch_output = ConvertLUT(batch_input, interval, weight)
                results = batch_output.cpu().data.numpy().astype(np.int8)
                outputs += [results]
        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def ConvertLUT(batch, interval, weight):
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1
        floor_batch = torch.div(batch, q, rounding_mode='floor')
        floor_batch_a1 = floor_batch[:, :, 0, 0].flatten()
        floor_batch_b1 = floor_batch[:, :, 0, 1].flatten()
        floor_batch_c1 = floor_batch[:, :, 1, 0].flatten()
        floor_batch_d1 = floor_batch[:, :, 1, 1].flatten()
        output = weight[floor_batch_a1 * L * L * L + floor_batch_b1 * L * L + floor_batch_c1 * L + floor_batch_d1]

        return output

    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    # main
    modes = [i for i in opt.stage1_modes]
    stages = opt.stages
    model_G = model(lut_folder=opt.lutDir, stages=stages, modes=modes, lutName=opt.lutName, upscale=opt.scale, interval=opt.interval, conversion=opt.conversion).cuda()

    input_tensor = get_input_tensor(opt)
    for mode in modes:
        if opt.cd == 'xyzt':
            input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
        elif opt.cd == 'xyz':
            input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
        elif opt.cd == 'xy':
            input_tensor_c1 = compress_lut(opt, input_tensor)
        else:
            raise ValueError
        input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

        # conv1
        key = "s{}c0_{}".format(1, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv2
        key = "s{}c0_{}".format(2, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv3
        key = "s{}c0_{}".format(3, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 3, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 3, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv4
        key = "s{}c0_{}".format(4, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv6
        for c in range(4):
            key = "s{}c{}_{}".format(6, c, mode)
            key = "weight_" + key
            module = getattr(model_G, key)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

    # conv5
    key = "s{}_channel".format(5)
    key = "weight_" + key
    module = getattr(model_G, key)
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_lut_ilf_compressed(input_tensor, lut_path, opt.interval, module)

def transfer_regular_lut_RFd1_to_compact_lut_together(opt, model):
    def save_lut_ilf_compressed(x, lut_path, interval, weight):
        B = x.size(0) // 100
        weight = weight * 127
        weight = round_func(weight)
        weight = torch.clamp(weight, -127, 127)
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]
                batch_output = ConvertLUT(batch_input, interval, weight)
                results = batch_output.cpu().data.numpy().astype(np.int8)
                outputs += [results]
        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def save_lut_ilf_compressed_3D(x, lut_path, interval, weight):
        B = x.size(0) // 100
        weight = weight * 127
        weight = round_func(weight)
        weight = torch.clamp(weight, -127, 127)
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]
                batch_output = ConvertLUT_3D(batch_input, interval, weight)
                results = batch_output.cpu().data.numpy().astype(np.int8)
                outputs += [results]
        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def ConvertLUT(batch, interval, weight):
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1
        floor_batch = torch.div(batch, q, rounding_mode='floor')
        floor_batch_a1 = floor_batch[:, :, 0, 0].flatten()
        floor_batch_b1 = floor_batch[:, :, 0, 1].flatten()
        floor_batch_c1 = floor_batch[:, :, 1, 0].flatten()
        floor_batch_d1 = floor_batch[:, :, 1, 1].flatten()
        output = weight[floor_batch_a1 * L * L * L + floor_batch_b1 * L * L + floor_batch_c1 * L + floor_batch_d1]

        return output

    def ConvertLUT_3D(batch, interval, weight):
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1
        floor_batch = torch.div(batch, q, rounding_mode='floor')
        floor_batch_a1 = floor_batch[:, :, 0, 0].flatten()
        floor_batch_b1 = floor_batch[:, :, 0, 1].flatten()
        floor_batch_c1 = floor_batch[:, :, 0, 2].flatten()
        output = weight[floor_batch_a1 * L * L + floor_batch_b1 * L + floor_batch_c1]

        return output

    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    # main
    modes = [i for i in opt.stage1_modes]
    stages = opt.stages
    model_G = model(loadIter=opt.loadIter, lut_folder=opt.lutDir, stages=stages, modes=modes, lutName=opt.lutName, upscale=opt.scale, interval=opt.interval, conversion=opt.conversion).cuda()

    input_tensor = get_input_tensor(opt)
    input_tensor_3D = get_input_tensor_3D(opt)

    for mode in modes:
        if opt.cd == 'xyzt':
            input_tensor_c1 = compress_lut_xyzt(opt, input_tensor)
        elif opt.cd == 'xyz':
            input_tensor_c1 = compress_lut_xyz(opt, input_tensor)
        elif opt.cd == 'xy':
            input_tensor_c1 = compress_lut(opt, input_tensor)
        else:
            raise ValueError
        input_tensor_c2 = compress_lut_larger_interval(opt, input_tensor)

        # conv1
        key = "s{}c0_{}".format(1, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 1, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv2
        key = "s{}c0_{}".format(2, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 2, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv3
        key = "s{}c0_{}".format(3, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1_iter{}.npy'.format(opt.lutName, 3, mode, opt.loadIter))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2_iter{}.npy'.format(opt.lutName, 3, mode, opt.loadIter))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv4
        key = "s{}c0_{}".format(4, mode)
        key = "weight_" + key
        module = getattr(model_G, key)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress1.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}_compress2.npy'.format(opt.lutName, 4, mode))
        save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

        # conv6
        for c in range(3):
            key = "s{}c{}_{}".format(6, c, mode)
            key = "weight_" + key
            module = getattr(model_G, key)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress1.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c1, lut_path, opt.interval, module)
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}_compress2.npy'.format(opt.lutName, 6, c, mode))
            save_lut_ilf_compressed(input_tensor_c2, lut_path, opt.interval, module)

    # conv5
    key = "s{}_channel".format(5)
    key = "weight_" + key
    module = getattr(model_G, key)
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_lut_ilf_compressed_3D(input_tensor_3D, lut_path, opt.interval, module)

if __name__ == "__main__":
    opt = TestOptions().parse()
    transfer_together = True
    if transfer_together:
        base_path = r"..\output\expr\LUT-pth"
        targer_path = r"..\output\expr\LUT-Transfer"
        if os.path.exists(base_path):
            files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
            file_count = len(files)
            opt.expDir = os.path.join(os.path.join(targer_path), '{}{}i{}'.format(opt.cd, opt.dw, opt.si))
            opt.lutDir = os.path.join(base_path)
            if not os.path.exists(opt.expDir):
                os.makedirs(opt.expDir)
            model = Model.LUT_ILF_Regular_LUT
            transfer_regular_lut_to_compact_lut_together(opt, model)