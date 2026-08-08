import os
import sys

import numpy as np

sys.path.insert(0, "../")
from common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "c": 2, "t": 2, "f": 2, "b": 2, "k": 2, "g": 2, "e": 3, "h": 3, "o": 3, "p": 3, "i": 3, "m": 3, "q": 3, "l": 3, "z": 3}

def round_func(input):
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

def identity(input):
    return input

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=1, output_quant=False, modes=['s'], nf=64, weight=True, ps_error=False):
        super(ConvBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.module_dict = dict()
        self.upscale = scale
        self.weight = weight
        self.output_quant = output_quant

        if self.weight:
            init_alphas = torch.ones([1 * out_c, len(modes) * in_c]) * 1 / len(modes)
            self.alphas = torch.nn.Parameter(init_alphas)

        if scale is None or 1:
            scale_factor = 1
        else:
            scale_factor = scale ** 2

        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = LUTConv('{}x{}'.format(mode.upper(), 'N'), nf=nf, out_c=out_c * scale_factor, stride=1)
        self.module_dict = nn.ModuleDict(self.module_dict)

        if ps_error:
            if scale is None:
                self.pixel_shuffle = identity
            else:
                self.pixel_shuffle = nn.PixelShuffle(scale)
        else:
            if scale is None or 1:
                self.pixel_shuffle = identity
            else:
                self.pixel_shuffle = nn.PixelShuffle(scale)

    def get_probs(self):
        with torch.no_grad():
            probs = []
            if self.weight:
                if self.out_c == 2 and self.in_c == 1:
                    for c in range(self.out_c):
                        prob = F.softmax(self.alphas[c, 0:len(self.modes)], dim=0)
                        probs.append(prob)
                elif self.out_c == 1 and self.in_c == 3:
                    for c in range(self.in_c):
                        prob = F.softmax(self.alphas[0, c * len(self.modes):(c + 1) * len(self.modes)], dim=0)
                        probs.append(prob)
                else:
                    prob = F.softmax(self.alphas[0, 0:len(self.modes)], dim=0)
                    probs.append(prob)
            return probs

    def forward(self, x):
        # weight init
        if self.weight:
            probs = []
            if self.out_c == 2 and self.in_c == 1:
                for c in range(self.out_c):
                    prob = F.softmax(self.alphas[c, 0:len(self.modes)], dim=0)
                    probs.append(prob)
            elif self.out_c == 1 and self.in_c == 3:
                for c in range(self.in_c):
                    prob = F.softmax(self.alphas[0, c*len(self.modes):(c+1)*len(self.modes)], dim=0)
                    probs.append(prob)
            else:
                prob = F.softmax(self.alphas[0, 0:len(self.modes)], dim=0)
                probs.append(prob)

        # prediction
        x_out = torch.zeros((x.size(0), self.out_c, x.size(2), x.size(3))).to(x.device)
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            for i, mode in enumerate(self.modes):
                pred = 0
                pad = mode_pad_dict[mode]
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.tanh(torch.rot90(self.pixel_shuffle(
                        sub_module(F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate'))),
                        (4 - r) % 4, [2, 3])) * 127)
                if self.weight:
                    if self.out_c == 2 and self.in_c == 1:
                        x_out[:, 0:1, :, :] += (probs[0])[i] * pred[:, 0:1, :, :]
                        x_out[:, 1:, :, :] += (probs[1])[i] * pred[:, 1:, :, :]
                    elif self.out_c == 1 and self.in_c == 3:
                        x_out += (probs[c])[i] * pred
                    else:
                        x_out += (probs[c])[i] * pred
                else:
                    x_out += pred

        # quantization
        if self.output_quant:
            avg_factor = len(self.modes) * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            x = x_out / self.in_c

        return x

class ConvBlock_s2(nn.Module):
    def __init__(self, in_c, out_c, scale=1, output_quant=False, modes=['s'], nf=64, weight=True, ps_error=False):
        super(ConvBlock_s2, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.module_dict = dict()
        self.upscale = scale
        self.weight = weight
        self.output_quant = output_quant

        if self.weight:
            init_alphas = torch.ones([1 * out_c, len(modes) * in_c]) * 1 / len(modes)
            self.alphas = torch.nn.Parameter(init_alphas)

        if scale is None or 1:
            scale_factor = 1
        else:
            scale_factor = scale ** 2

        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = LUTConv('{}x{}'.format(mode.upper(), 'N'), nf=nf, out_c=out_c * scale_factor, stride=2)
        self.module_dict = nn.ModuleDict(self.module_dict)

        if ps_error:
            if scale is None:
                self.pixel_shuffle = identity
            else:
                self.pixel_shuffle = nn.PixelShuffle(scale)
        else:
            if scale is None or 1:
                self.pixel_shuffle = identity
            else:
                self.pixel_shuffle = nn.PixelShuffle(scale)

    def get_probs(self):
        with torch.no_grad():
            probs = []
            if self.weight:
                if self.out_c == 2 and self.in_c == 1:
                    for c in range(self.out_c):
                        prob = F.softmax(self.alphas[c, 0:len(self.modes)], dim=0)
                        probs.append(prob)
                elif self.out_c == 1 and self.in_c == 3:
                    for c in range(self.in_c):
                        prob = F.softmax(self.alphas[0, c * len(self.modes):(c + 1) * len(self.modes)], dim=0)
                        probs.append(prob)
                else:
                    prob = F.softmax(self.alphas[0, 0:len(self.modes)], dim=0)
                    probs.append(prob)
            return probs

    def forward(self, x):
        # weight init
        if self.weight:
            probs = []
            if self.out_c == 2 and self.in_c == 1:
                for c in range(self.out_c):
                    prob = F.softmax(self.alphas[c, 0:len(self.modes)], dim=0)
                    probs.append(prob)
            elif self.out_c == 1 and self.in_c == 3:
                for c in range(self.in_c):
                    prob = F.softmax(self.alphas[0, c*len(self.modes):(c+1)*len(self.modes)], dim=0)
                    probs.append(prob)
            else:
                prob = F.softmax(self.alphas[0, 0:len(self.modes)], dim=0)
                probs.append(prob)

        # prediction
        x_out = torch.zeros((x.size(0), self.out_c, x.size(2) // 2, x.size(3) // 2)).to(x.device)
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            for i, mode in enumerate(self.modes):
                pred = 0
                pad = mode_pad_dict[mode]
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.tanh(torch.rot90(self.pixel_shuffle(
                        sub_module(F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate'))),
                        (4 - r) % 4, [2, 3])) * 127)
                if self.weight:
                    if self.out_c == 2 and self.in_c == 1:
                        x_out[:, 0:1, :, :] += (probs[0])[i] * pred[:, 0:1, :, :]
                        x_out[:, 1:, :, :] += (probs[1])[i] * pred[:, 1:, :, :]
                    elif self.out_c == 1 and self.in_c == 3:
                        x_out += (probs[c])[i] * pred
                    else:
                        x_out += (probs[c])[i] * pred
                else:
                    x_out += pred

        # quantization
        if self.output_quant:
            avg_factor = len(self.modes) * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            x = x_out / self.in_c

        return x

class LUT_ILF_Net(nn.Module):
    def __init__(self, nf=64, scale=1, stage1_modes=['s'], stage2_modes=['s'], stage3_modes=['s'], stage4_modes=['s'], stages=4, weight=False, ps_error=False):
        super(LUT_ILF_Net, self).__init__()
        self.upscale = scale
        self.weight = weight
        self.stage1_modes = stage1_modes
        self.stage2_modes = stage2_modes
        self.stage3_modes = stage3_modes
        self.stage4_modes = stage4_modes

        self.convblock1 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock2 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage2_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock3 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage2_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock4 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage3_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.ChannelConv = LUTChannelUnit(in_c=4, out_c=4, mode='1x1', nf=nf)
        self.upblock = ConvBlock(4, 1, scale=scale, output_quant=False, modes=stage4_modes, nf=nf, weight=weight, ps_error=ps_error)


    def forward(self, x, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        x = self.convblock1(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        x = self.convblock2(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage2_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        x = self.convblock3(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage2_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        x = self.convblock4(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage3_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        x = self.upblock(x)
        if self.weight:
            avg_factor, bias, norm = 1, 0, 1
        else:
            avg_factor, bias, norm = len(self.stage4_modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H, self.upscale * W))

        return x

class LUT_ILF_Net_RDd1(nn.Module):
    def __init__(self, nf=64, scale=1, stage1_modes=['s'], stage2_modes=['s'], stage3_modes=['s'], stage4_modes=['s'], stages=4, weight=False, ps_error=False):
        super(LUT_ILF_Net_RDd1, self).__init__()
        self.upscale = scale
        self.weight = weight
        self.stage1_modes = stage1_modes
        self.stage2_modes = stage2_modes
        self.stage3_modes = stage3_modes
        self.stage4_modes = stage4_modes

        self.convblock1 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock2 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage2_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock3 = ConvBlock(1, 2, scale=None, output_quant=False, modes=stage2_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.convblock4 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage3_modes, nf=nf, weight=weight, ps_error=ps_error)
        self.ChannelConv = LUTChannelUnit(in_c=3, out_c=3, mode='1x1', nf=nf)
        self.upblock = ConvBlock(3, 1, scale=scale, output_quant=False, modes=stage4_modes, nf=nf, weight=weight, ps_error=ps_error)


    def forward(self, x, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        x = self.convblock1(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        x = self.convblock2(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage2_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        x = self.convblock4(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage3_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        x = self.upblock(x)
        if self.weight:
            avg_factor, bias, norm = 1, 0, 1
        else:
            avg_factor, bias, norm = len(self.stage4_modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H, self.upscale * W))

        return x

class LUT_ILF_Net_cc_u_alf(nn.Module):
    def __init__(self, nf=64, scale=1, stage1_modes=['s'], weight=False, ps_error=False):
        super(LUT_ILF_Net_cc_u_alf, self).__init__()
        self.upscale = scale
        self.weight = weight
        self.stage1_modes = stage1_modes

        self.downblock = ConvBlock_s2(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                      ps_error=ps_error)
        self.ChannelConv = LUTChannelUnit(in_c=3, out_c=3, mode='1x1', nf=nf)
        self.upblock = ConvBlock(3, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                 ps_error=ps_error)
        self.convblock1 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                    ps_error=ps_error)
        self.convblock2 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                    ps_error=ps_error)
        self.convblock3 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                    ps_error=ps_error)
        self.convblock4 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                    ps_error=ps_error)
        self.convblock5 = ConvBlock(1, 1, scale=None, output_quant=False, modes=stage1_modes, nf=nf, weight=weight,
                                    ps_error=ps_error)

    def forward(self, x, x_u, x_v, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))
        B_u, C_u, H_u, W_u = x_u.size()
        x_u = x_u.reshape((B_u * C_u, 1, H_u, W_u))
        x_v = x_v.reshape((B_u * C_u, 1, H_u, W_u))

        x = self.downblock(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list = []
        refine_list.append(x)
        refine_list.append(x_u)
        refine_list.append(x_v)

        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        x = self.upblock(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        x = self.convblock1(x)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        x = self.convblock2(x)
        if self.weight:
            avg_factor, bias, norm = 1, 0, 1
        else:
            avg_factor, bias, norm = len(self.stage1_modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        x_u = self.convblock3(x_u)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x_u = round_func(torch.clamp((x_u / avg_factor) + bias, 0, 255)) / norm

        x_u = self.convblock4(x_u)
        if self.weight:
            avg_factor, bias, norm = 1 * 4, 127, 255.0
        else:
            avg_factor, bias, norm = len(self.stage1_modes) * 4, 127, 255.0
        x_u = round_func(torch.clamp((x_u / avg_factor) + bias, 0, 255)) / norm

        x_u = self.convblock5(x_u)
        if self.weight:
            avg_factor, bias, norm = 1, 0, 1
        else:
            avg_factor, bias, norm = len(self.stage1_modes), 0, 1
        x_u = round_func((x_u / avg_factor) + bias)

        x = x + x_u

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H_u, self.upscale * W_u))

        return x

class LUT_ILF_Compressed_LUT(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, phase = 'train'):
        super(LUT_ILF_Compressed_LUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.d = diagonal_width
        L = 2 ** (8 - interval) + 1
        self.compression_type = compressed_dimensions
        # self.diagonal_width = diagonal_width
        self.sampling_interval = sampling_interval

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress1".format(1, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path)
                lut_arr = lut_arr.reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress2".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress1".format(2, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress2".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress1".format(3, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress2".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress1".format(4, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress2".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                if compressed_dimensions=='xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        if phase == 'train':
            lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        else:
            lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out


    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a1]
        k11 = self.ref2index[img_a2, img_b2 - img_a1]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag


    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag


    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag


    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c * upscale * upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all
        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        # out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
        #                    img_a1.shape[3], out_c*upscale*upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        # sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        # out_all = out.reshape(sz, -1)
        # out = out_all[~index_flag]

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0,sz1,sz2,sz3, out_c,upscale, upscale))
        out_all = out_all.permute(0,1,4,2,5,3,6).reshape(
            (sz0, sz1*out_c, sz2 * upscale, sz3 * upscale))
        return out_all


    def InterpTorchBatch_channel(self, weight,out_c, img_in):

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1, img_b1, img_c1, img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1, img_b1, img_c1, img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in%q,4,1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                                   img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape(
            (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))
        return out


    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C,1,H,W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2,2,2,1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale =1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s]==2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight,4,x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))
        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:,c:c+1,:,:]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B,C,H*self.upscale,W*self.upscale))

        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Regular_LUT(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(LUT_ILF_Regular_LUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}".format(6, c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

        return out

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        out_c_list = [2, 2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight, 4, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6,c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Regular_LUT_RFd1(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(LUT_ILF_Regular_LUT_RFd1, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 3, mode))
            # key = "s{}c0_{}".format(3, mode)
            # lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(3):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}".format(6, c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 3)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

        return out

    def InterpTorchBatch_Channel_3D(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_a1,img_b1,img_c1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 3, 1)
        else:
            img_a1,img_b1,img_c1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 3, 1)

        fa,fb,fc = torch.chunk(img_in % q, 3, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1

        p000 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p001 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p010 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p100 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p011 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p101 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p110 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p111 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p000 = p000.reshape(sz, -1)
        p001 = p001.reshape(sz, -1)
        p010 = p010.reshape(sz, -1)
        p100 = p100.reshape(sz, -1)
        p011 = p011.reshape(sz, -1)
        p101 = p101.reshape(sz, -1)
        p110 = p110.reshape(sz, -1)
        p111 = p111.reshape(sz, -1)

        fa = fa.reshape(-1, 1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        out = (p000 * (q - fa) * (q - fb) * (q - fc) + p001 * (q - fa) * (q - fb) * fc +
               p010 * (q - fa) * fb * (q - fc) + p011 * (q - fa) * fb * fc +
               p100 * fa * (q - fb) * (q - fc) + p101 * fa * (q - fb) * fc +
               p110 * fa * fb * (q - fc) + p111 * fa * fb * fc)

        out = out / (q ** 3)
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        out_c_list = [2, 2, 1]
        refine_list = []

        # conv1~3
        for s in range(3):
            stage = s+1
            if s==2:
                stage += 1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)

        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_Channel_3D(weight, 3, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(3):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6, c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 3
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Regular_LUT_Test(nn.Module):
    def __init__(self, loadIter, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(LUT_ILF_Regular_LUT_Test, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.loadIter = loadIter

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 3, mode, self.loadIter))
            key = "s{}c0_{}".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_{:06d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}".format(6, c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel_{:06d}.npy'.format(lutName, 5, self.loadIter))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

        return out

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        out_c_list = [2, 2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight, 4, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6,c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Regular_LUT_RFd1_Test(nn.Module):
    def __init__(self, loadIter, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(LUT_ILF_Regular_LUT_RFd1_Test, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.loadIter = loadIter

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 3, mode))
            # key = "s{}c0_{}".format(3, mode)
            # lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_{:06d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(3):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_{:06d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}".format(6, c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel_{:06d}.npy'.format(lutName, 5, self.loadIter))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 3)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

        return out

    def InterpTorchBatch_Channel_3D(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_a1,img_b1,img_c1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 3, 1)
        else:
            img_a1,img_b1,img_c1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 3, 1)

        fa,fb,fc = torch.chunk(img_in % q, 3, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1

        p000 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p001 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p010 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p100 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p011 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p101 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p110 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p111 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p000 = p000.reshape(sz, -1)
        p001 = p001.reshape(sz, -1)
        p010 = p010.reshape(sz, -1)
        p100 = p100.reshape(sz, -1)
        p011 = p011.reshape(sz, -1)
        p101 = p101.reshape(sz, -1)
        p110 = p110.reshape(sz, -1)
        p111 = p111.reshape(sz, -1)

        fa = fa.reshape(-1, 1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        out = (p000 * (q - fa) * (q - fb) * (q - fc) + p001 * (q - fa) * (q - fb) * fc +
               p010 * (q - fa) * fb * (q - fc) + p011 * (q - fa) * fb * fc +
               p100 * fa * (q - fb) * (q - fc) + p101 * fa * (q - fb) * fc +
               p110 * fa * fb * (q - fc) + p111 * fa * fb * fc)

        out = out / (q ** 3)
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        out_c_list = [2, 2, 1]
        refine_list = []

        # conv1~3
        for s in range(3):
            stage = s+1
            if s==2:
                stage += 1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)

        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_Channel_3D(weight, 3, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(3):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6, c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 3
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x

class _LUT_ILF_Chroma_LUT_Base(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(_LUT_ILF_Chroma_LUT_Base, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # downconv
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # upconv
            for c in range(3):
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}".format(6, c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 3)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c, upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

        return out

    def InterpTorchBatch_Channel_3D(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_a1,img_b1,img_c1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 3, 1)
        else:
            img_a1,img_b1,img_c1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 3, 1)

        fa,fb,fc = torch.chunk(img_in % q, 3, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1

        p000 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p001 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p010 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p100 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p011 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p101 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p110 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p111 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p000 = p000.reshape(sz, -1)
        p001 = p001.reshape(sz, -1)
        p010 = p010.reshape(sz, -1)
        p100 = p100.reshape(sz, -1)
        p011 = p011.reshape(sz, -1)
        p101 = p101.reshape(sz, -1)
        p110 = p110.reshape(sz, -1)
        p111 = p111.reshape(sz, -1)

        fa = fa.reshape(-1, 1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        out = (p000 * (q - fa) * (q - fb) * (q - fc) + p001 * (q - fa) * (q - fb) * fc +
               p010 * (q - fa) * fb * (q - fc) + p011 * (q - fa) * fb * fc +
               p100 * fa * (q - fb) * (q - fc) + p101 * fa * (q - fb) * fc +
               p110 * fa * fb * (q - fc) + p111 * fa * fb * fc)

        out = out / (q ** 3)
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def forward(self, x, x_u, x_v, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W)) * 255.0
        B_u, C_u, H_u, W_u = x_u.size()
        x_u = x_u.reshape((B_u * C_u, 1, H_u, W_u)) * 255.0
        x_v = x_v.reshape((B_u * C_u, 1, H_u, W_u)) * 255.0

        # downblock-Y
        pred = 0
        for mode in self.modes:
            pad = mode_pad_dict[mode]
            key = "s{}c0_{}".format(str(3), mode)
            weight = getattr(self, "weight_" + key)
            scale = 1
            for r in [0, 1, 2, 3]:
                x_rot = F.pad(torch.rot90(x, r, [2, 3]),
                              (0, pad, 0, pad), mode='replicate')
                pred_rot = self.InterpTorchBatch(weight, scale, 1, mode, x_rot, pad)

                # The corresponding network block is LUTConv(stride=2).
                # Apply the stride in the rotated coordinate system before
                # rotating the result back; downsampling after rotation fusion
                # selects a different sampling phase for the 90/270 branches.
                pred_rot = pred_rot[..., ::2, ::2].contiguous()
                pred += torch.rot90(pred_rot, (4 - r) % 4, [2, 3])
                pred = self.round_func(pred)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))

        if x.shape[-2:] != x_u.shape[-2:]:
            raise ValueError("Downsampled Y shape {} does not match UV shape {}"
                             .format(tuple(x.shape[-2:]), tuple(x_u.shape[-2:])))

        # channel-conv
        refine_list = []
        refine_list.append(x)
        refine_list.append(x_u)
        refine_list.append(x_v)
        x = torch.cat(refine_list, dim=1)
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_Channel_3D(weight, 3, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # upconv
        pred = 0
        for c in range(3):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6, c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 3
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H_u*self.upscale, W_u*self.upscale))

        # conv1
        pred = 0
        for mode in self.modes:
            pad = mode_pad_dict[mode]
            key = "s{}c0_{}".format(str(1), mode)
            weight = getattr(self, "weight_" + key)
            scale = 1
            for r in [0, 1, 2, 3]:
                pred += torch.rot90(
                    self.InterpTorchBatch(weight, scale, 1, mode, F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                pred = self.round_func(pred)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))

        # conv2
        pred = 0
        for mode in self.modes:
            pad = mode_pad_dict[mode]
            key = "s{}c0_{}".format(str(2), mode)
            weight = getattr(self, "weight_" + key)
            scale = 1
            for r in [0, 1, 2, 3]:
                pred += torch.rot90(
                    self.InterpTorchBatch(weight, scale, 1, mode, F.pad(torch.rot90(x, r, [
                        2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                pred = self.round_func(pred)
        avg_factor, bias = len(self.modes), 0
        x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
        
        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Net_cc_u_alf_LUT(_LUT_ILF_Chroma_LUT_Base):
    """Regular-LUT implementation of LUT_ILF_Net_cc_u_alf.

    Stages 1/2/3/5/6 keep the base UV LUT layout.  Stages 7/8/9
    represent convblock3/4/5 on the parallel target-chroma ALF branch.
    """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, conversion):
        super(LUT_ILF_Net_cc_u_alf_LUT, self).__init__(
            lut_folder=lut_folder,
            stages=stages,
            modes=modes,
            lutName=lutName,
            upscale=upscale,
            interval=interval,
            conversion=conversion,
        )

        for mode in modes:
            for stage in (7, 8, 9):
                lut_path = os.path.join(
                    lut_folder, '{}_s{}c0_{}.npy'.format(lutName, stage, mode)
                )
                key = "s{}c0_{}".format(stage, mode)
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
                self.register_parameter(
                    name="weight_" + key,
                    param=torch.nn.Parameter(torch.Tensor(lut_arr)),
                )

    def _spatial_stage(self, x, stage, final_stage=False, stride=1):
        pred = 0
        for mode in self.modes:
            pad = mode_pad_dict[mode]
            key = "s{}c0_{}".format(stage, mode)
            weight = getattr(self, "weight_" + key)

            for rotation in (0, 1, 2, 3):
                x_rot = F.pad(
                    torch.rot90(x, rotation, [2, 3]),
                    (0, pad, 0, pad),
                    mode='replicate',
                )
                pred_rot = self.InterpTorchBatch(
                    weight, 1, 1, mode, x_rot, pad
                )
                if stride == 2:
                    pred_rot = pred_rot[..., ::2, ::2].contiguous()
                pred += torch.rot90(
                    pred_rot, (4 - rotation) % 4, [2, 3]
                )
                pred = self.round_func(pred)

        if final_stage:
            # convblock2/convblock5 are signed branch outputs.  Do not
            # clamp either branch before their residual addition.
            return self.round_func(pred / len(self.modes))

        avg_factor = len(self.modes) * 4
        return self.round_func(torch.clamp(pred / avg_factor + 127, 0, 255))

    def _channel_stage(self, downsampled_y, x_u, x_v):
        x = torch.cat([downsampled_y, x_u, x_v], dim=1)
        weight = getattr(self, "weight_s5_channel")
        x = self.InterpTorchBatch_Channel_3D(weight, 3, x)
        return self.round_func(torch.clamp(x + 127, 0, 255))

    def _up_stage(self, x):
        pred = 0
        for channel in range(3):
            x_c = x[:, channel:channel + 1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s6c{}_{}".format(channel, mode)
                weight = getattr(self, "weight_" + key)

                for rotation in (0, 1, 2, 3):
                    x_rot = F.pad(
                        torch.rot90(x_c, rotation, [2, 3]),
                        (0, pad, 0, pad),
                        mode='replicate',
                    )
                    pred_rot = self.InterpTorchBatch(
                        weight, self.upscale, 1, mode, x_rot, pad
                    )
                    pred += torch.rot90(
                        pred_rot, (4 - rotation) % 4, [2, 3]
                    )
                    pred = self.round_func(pred)

        pred = pred / 3
        avg_factor = len(self.modes) * 4
        return self.round_func(torch.clamp(pred / avg_factor + 127, 0, 255))

    def forward(self, x, x_u, x_v, phase='train'):
        batch, channels, height, width = x.shape
        x = x.reshape((batch * channels, 1, height, width)) * 255.0

        batch_u, channels_u, height_u, width_u = x_u.shape
        x_u = x_u.reshape((batch_u * channels_u, 1, height_u, width_u)) * 255.0
        x_v = x_v.reshape((batch_u * channels_u, 1, height_u, width_u)) * 255.0

        # Main cross-component branch: Y -> down -> [Y, U, V] -> stages 6/1/2.
        main = self._spatial_stage(x, stage=3, stride=2)
        if main.shape[-2:] != x_u.shape[-2:]:
            raise ValueError(
                "Downsampled Y shape {} does not match UV shape {}".format(
                    tuple(main.shape[-2:]), tuple(x_u.shape[-2:])
                )
            )
        main = self._channel_stage(main, x_u, x_v)
        main = self._up_stage(main)
        main = self._spatial_stage(main, stage=1)
        main = self._spatial_stage(main, stage=2, final_stage=True)

        # Parallel target-chroma ALF branch: stages 7/8/9 correspond to
        # convblock3/4/5. V reuses this class by passing (Y, V, U).
        alf = self._spatial_stage(x_u, stage=7)
        alf = self._spatial_stage(alf, stage=8)
        alf = self._spatial_stage(alf, stage=9, final_stage=True)

        output = main + alf
        output = output.reshape((batch, channels, height_u * self.upscale,
                                 width_u * self.upscale))
        if phase == 'train':
            output = output / 255.0

        return output

class LUT_ILF_Compact_LUT(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, conversion):
        super(LUT_ILF_Compact_LUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.d = diagonal_width
        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress1".format(1, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress2".format(1, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress1".format(2, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress2".format(2, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress1".format(3, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress2".format(3, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress1".format(4, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress2".format(4, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                key = "weight_" + key
                if compressed_dimensions == 'xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                key = "weight_" + key
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        key = "weight_" + key
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3],
                           out_c * upscale * upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
        out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            (sz0, sz1 * out_c, sz2 * upscale, sz3 * upscale))

        return out_all

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a2]
        k11 = self.ref2index[img_a2, img_b2 - img_a2]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))

        out = torch.zeros((img_a1.shape[0], out_c * upscale * upscale), dtype=weight_c1.dtype).to(
            device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c * upscale * upscale))

        out = out / q

        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2, 2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "weight_" + "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, key)
                key = "weight_" + "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight, 4, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x
    
class LUT_ILF_Net_cc_u_alf_Compact_LUT(LUT_ILF_Net_cc_u_alf_LUT):
    """Compact-LUT runtime for the cascaded U/V ``cc_u_alf`` pipeline.

    Spatial stages 1/2/3/7/8/9 and the three stage-6 tables use the
    diagonal-region ``compress1`` plus coarse ``compress2`` representation.
    Stage 5 remains a regular 3-D [Y, target chroma, other chroma] LUT.
    The inherited chroma forward path preserves the stride-2 Y indexing and
    the signed main/ALF residual addition used by the regular-LUT runtime.
    """

    # Reuse the established compact interpolation implementation. These
    # methods depend only on attributes initialized below and round_func from
    # the regular chroma base class.
    InterpTorchBatch = LUT_ILF_Compact_LUT.InterpTorchBatch
    InterpTorchBatch_compress1_xy = LUT_ILF_Compact_LUT.InterpTorchBatch_compress1_xy
    InterpTorchBatch_compress1_xyz = LUT_ILF_Compact_LUT.InterpTorchBatch_compress1_xyz
    InterpTorchBatch_compress1_xyzt = LUT_ILF_Compact_LUT.InterpTorchBatch_compress1_xyzt

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval,
                 compressed_dimensions, diagonal_width, sampling_interval,
                 conversion=True, load_iter=0):
        nn.Module.__init__(self)
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.d = diagonal_width
        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval

        if compressed_dimensions not in ('xy', 'xyz', 'xyzt'):
            raise ValueError(
                'Wrong Compressed Dimensions (should be xy, xyz, or xyzt).'
            )
        if sampling_interval <= interval:
            raise ValueError(
                'sampling_interval must be larger than interval: {} <= {}'
                .format(sampling_interval, interval)
            )

        index_name = 'ref2index_{}{}i{}.npy'.format(
            compressed_dimensions, diagonal_width, sampling_interval
        )
        index_path = os.path.join(lut_folder, index_name)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                'Compact LUT index file not found: {}'.format(index_path)
            )
        ref2index = torch.from_numpy(np.load(index_path)).type(torch.int64)
        self.register_buffer('ref2index', ref2index)

        L = 2 ** (8 - interval) + 1

        def resolve_path(stem):
            if load_iter > 0:
                iter_path = os.path.join(
                    lut_folder, '{}_{:06d}.npy'.format(stem, load_iter)
                )
                if os.path.isfile(iter_path):
                    return iter_path
                raise FileNotFoundError(
                    'Compact LUT iteration file not found: {}'.format(iter_path)
                )
            path = os.path.join(lut_folder, stem + '.npy')
            if not os.path.isfile(path):
                raise FileNotFoundError('Compact LUT file not found: {}'.format(path))
            return path

        def register_compact(key, output_channels):
            stem1 = '{}_{}_compress1'.format(lutName, key)
            lut_arr = np.load(resolve_path(stem1)).astype(np.float32) / 127.0
            if compressed_dimensions == 'xy':
                lut_arr = lut_arr.reshape((-1, L * L, output_channels))
            elif compressed_dimensions == 'xyz':
                lut_arr = lut_arr.reshape((-1, L, output_channels))
            else:
                lut_arr = lut_arr.reshape((-1, output_channels))
            self.register_parameter(
                name='weight_{}_compress1'.format(key),
                param=torch.nn.Parameter(torch.from_numpy(lut_arr)),
            )

            stem2 = '{}_{}_compress2'.format(lutName, key)
            lut_arr = np.load(resolve_path(stem2)).reshape(
                (-1, output_channels)
            ).astype(np.float32) / 127.0
            self.register_parameter(
                name='weight_{}_compress2'.format(key),
                param=torch.nn.Parameter(torch.from_numpy(lut_arr)),
            )

        for mode in modes:
            for stage in (1, 2, 3, 7, 8, 9):
                register_compact('s{}c0_{}'.format(stage, mode), 1)
            for channel in range(3):
                register_compact(
                    's6c{}_{}'.format(channel, mode),
                    self.upscale * self.upscale,
                )

        channel_stem = '{}_s5_channel'.format(lutName)
        lut_arr = np.load(resolve_path(channel_stem)).reshape(
            (-1, 3)
        ).astype(np.float32) / 127.0
        self.register_parameter(
            name='weight_s5_channel',
            param=torch.nn.Parameter(torch.from_numpy(lut_arr)),
        )

    def _spatial_stage(self, x, stage, final_stage=False, stride=1):
        pred = 0
        for mode in self.modes:
            pad = mode_pad_dict[mode]
            key = 's{}c0_{}'.format(stage, mode)
            weight_c1 = getattr(self, 'weight_{}_compress1'.format(key))
            weight_c2 = getattr(self, 'weight_{}_compress2'.format(key))

            for rotation in (0, 1, 2, 3):
                x_rot = F.pad(
                    torch.rot90(x, rotation, [2, 3]),
                    (0, pad, 0, pad),
                    mode='replicate',
                )
                pred_rot = self.InterpTorchBatch(
                    weight_c1, weight_c2, 1, 1, mode, x_rot, pad
                )
                if stride == 2:
                    pred_rot = pred_rot[..., ::2, ::2].contiguous()
                pred += torch.rot90(
                    pred_rot, (4 - rotation) % 4, [2, 3]
                )
                pred = self.round_func(pred)

        if final_stage:
            return self.round_func(pred / len(self.modes))
        avg_factor = len(self.modes) * 4
        return self.round_func(torch.clamp(pred / avg_factor + 127, 0, 255))

    def _up_stage(self, x):
        pred = 0
        for channel in range(3):
            x_c = x[:, channel:channel + 1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = 's6c{}_{}'.format(channel, mode)
                weight_c1 = getattr(self, 'weight_{}_compress1'.format(key))
                weight_c2 = getattr(self, 'weight_{}_compress2'.format(key))

                for rotation in (0, 1, 2, 3):
                    x_rot = F.pad(
                        torch.rot90(x_c, rotation, [2, 3]),
                        (0, pad, 0, pad),
                        mode='replicate',
                    )
                    pred_rot = self.InterpTorchBatch(
                        weight_c1, weight_c2, self.upscale, 1,
                        mode, x_rot, pad,
                    )
                    pred += torch.rot90(
                        pred_rot, (4 - rotation) % 4, [2, 3]
                    )
                    pred = self.round_func(pred)

        pred = pred / 3
        avg_factor = len(self.modes) * 4
        return self.round_func(torch.clamp(pred / avg_factor + 127, 0, 255))


class LUT_ILF_Compact_LUT_RFd1(nn.Module):
    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, conversion):
        super(LUT_ILF_Compact_LUT_RFd1, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.d = diagonal_width
        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress1".format(1, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress2".format(1, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress1".format(2, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress2".format(2, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            # key = "s{}c0_{}_compress1".format(3, mode)
            # key = "weight_" + key
            # if compressed_dimensions == 'xy':
            #     lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            # elif compressed_dimensions == 'xyz':
            #     lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            # elif compressed_dimensions == 'xyzt':
            #     lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # else:
            #     raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            # self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            # key = "s{}c0_{}_compress2".format(3, mode)
            # key = "weight_" + key
            # lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress1".format(4, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress2".format(4, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(3):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                key = "weight_" + key
                if compressed_dimensions == 'xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                key = "weight_" + key
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        key = "weight_" + key
        lut_arr = np.load(lut_path).reshape((-1, 3)).astype(np.float32) / 127.0
        self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3],
                           out_c * upscale * upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
        out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            (sz0, sz1 * out_c, sz2 * upscale, sz3 * upscale))

        return out_all

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d * q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a2]
        k11 = self.ref2index[img_a2, img_b2 - img_a2]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c * upscale * upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c * upscale * upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c * upscale * upscale))

        out = torch.zeros((img_a1.shape[0], out_c * upscale * upscale), dtype=weight_c1.dtype).to(
            device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c * upscale * upscale))

        out = out / q

        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B * C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(3):
            stage = s + 1
            if s == 2:
                stage += 1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "weight_" + "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, key)
                key = "weight_" + "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, out_c_list[s], mode,
                                              F.pad(torch.rot90(x, r, [
                                                  2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4,
                        [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        # The RF-1 Compact runtime shares the unchanged 3-D Channel-LUT
        # interpolation with its Regular-LUT counterpart.
        x = LUT_ILF_Regular_LUT_RFd1.InterpTorchBatch_Channel_3D(
            self, weight, 3, x
        )
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(3):
            x_c = x[:, c:c + 1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
        pred = pred / 3
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H * self.upscale, W * self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x

class LUT_ILF_Compact_LUT_Test(nn.Module):
    def __init__(self, loadIter, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, conversion):
        super(LUT_ILF_Compact_LUT_Test, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.loadIter = loadIter
        self.d = diagonal_width
        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(1, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(1, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(2, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(2, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 3, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(3, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 3, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(3, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(4, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(4, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1_iter{:05d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                key = "weight_" + key
                if compressed_dimensions == 'xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2_iter{:05d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                key = "weight_" + key
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel_iter{:05d}.npy'.format(lutName, 5, self.loadIter))
        key = "s{}_channel".format(5)
        key = "weight_" + key
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3],
                           out_c * upscale * upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
        out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            (sz0, sz1 * out_c, sz2 * upscale, sz3 * upscale))

        return out_all

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a2]
        k11 = self.ref2index[img_a2, img_b2 - img_a2]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q

        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2, 2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "weight_" + "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, key)
                key = "weight_" + "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight, 4, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x
    
class LUT_ILF_Compact_LUT_RFd1_Test(nn.Module):
    def __init__(self, loadIter, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, conversion):
        super(LUT_ILF_Compact_LUT_RFd1_Test, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.conversion = conversion
        self.loadIter = loadIter
        self.d = diagonal_width
        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval
        L = 2 ** (8 - interval) + 1

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(self.compression_type, self.d, self.sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(1, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 1, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(1, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(2, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 2, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(2, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 3, mode, self.loadIter))
            # key = "s{}c0_{}_compress1".format(3, mode)
            # key = "weight_" + key
            # if compressed_dimensions == 'xy':
            #     lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            # elif compressed_dimensions == 'xyz':
            #     lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            # elif compressed_dimensions == 'xyzt':
            #     lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # else:
            #     raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            # self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 3, mode, self.loadIter))
            # key = "s{}c0_{}_compress2".format(3, mode)
            # key = "weight_" + key
            # lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            # self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1_iter{:05d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}_compress1".format(4, mode)
            key = "weight_" + key
            if compressed_dimensions == 'xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions == 'xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2_iter{:05d}.npy'.format(lutName, 4, mode, self.loadIter))
            key = "s{}c0_{}_compress2".format(4, mode)
            key = "weight_" + key
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(3):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1_iter{:05d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                key = "weight_" + key
                if compressed_dimensions == 'xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions == 'xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2_iter{:05d}.npy'.format(lutName, 6, c, mode, self.loadIter))
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                key = "weight_" + key
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel_iter{:05d}.npy'.format(lutName, 5, self.loadIter))
        key = "s{}_channel".format(5)
        key = "weight_" + key
        lut_arr = np.load(lut_path).reshape((-1, 3)).astype(np.float32) / 127.0
        self.register_parameter(name=key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            raise ValueError("Mode {} not implemented.".format(mode))

        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3],
                           out_c * upscale * upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all

        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
        out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            (sz0, sz1 * out_c, sz2 * upscale, sz3 * upscale))

        return out_all

    def InterpTorchBatch_channel(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        if self.conversion:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 4, 1)
        else:
            img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 4, 1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in % q, 4, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def InterpTorchBatch_channel_3D(self, weight, out_c, img_in):
        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval
        L = 2 ** (8 - interval) + 1

        if self.conversion:
            img_a1,img_b1,img_c1 = torch.chunk(torch.div(img_in, q, rounding_mode='floor').type(torch.int64), 3, 1)
        else:
            img_a1,img_b1,img_c1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64), 3, 1)

        fa,fb,fc = torch.chunk(img_in % q, 3, 1)

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1

        p000 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p001 = weight[img_a1.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p010 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p100 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p011 = weight[img_a1.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p101 = weight[img_a2.flatten() * L * L + img_b1.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p110 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p111 = weight[img_a2.flatten() * L * L + img_b2.flatten() * L + img_c2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p000 = p000.reshape(sz, -1)
        p001 = p001.reshape(sz, -1)
        p010 = p010.reshape(sz, -1)
        p100 = p100.reshape(sz, -1)
        p011 = p011.reshape(sz, -1)
        p101 = p101.reshape(sz, -1)
        p110 = p110.reshape(sz, -1)
        p111 = p111.reshape(sz, -1)

        fa = fa.reshape(-1, 1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        out = (p000 * (q - fa) * (q - fb) * (q - fc) + p001 * (q - fa) * (q - fb) * fc +
               p010 * (q - fa) * fb * (q - fc) + p011 * (q - fa) * fb * fc +
               p100 * fa * (q - fb) * (q - fc) + p101 * fa * (q - fb) * fc +
               p110 * fa * fb * (q - fc) + p111 * fa * fb * fc)

        out = out / (q ** 3)
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape((img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))

        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a2]
        k11 = self.ref2index[img_a2, img_b2 - img_a2]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q

        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        if self.conversion:
            img_abcd = torch.div(img_in, q, rounding_mode='floor').type(torch.int64)
        else:
            img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Use the corrected comparison order to avoid interpolation overflow.
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def forward(self, x, phase='train'):
        B, C, H, W = x.shape
        x = x.reshape((B*C, 1, H, W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2, 2, 1]
        refine_list = []
        # conv1~4
        for s in range(3):
            stage = s+1
            if s == 2:
                stage += 1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "weight_" + "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, key)
                key = "weight_" + "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, key)
                scale = 1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s] == 2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel_3D(weight, 3, x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(3):
            x_c = x[:, c:c+1, :, :]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6, c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6, c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 3
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B, C, H*self.upscale, W*self.upscale))
        if phase == 'train':
            x = x / 255.0

        return x
