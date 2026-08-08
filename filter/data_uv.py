import os
import random
import sys

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "../")  # run under the project directory
from common.utils import modcrop


BVI_DVC_size = {"A": [3840, 2176, 16], "B": [1920, 1088, 8], "C": [960, 544, 4], "D": [480, 272, 2]}
VVC_class = {"Ah":'Tango2_3840x2160,ParkRunning3_3840x2160,FoodMarket4_3840x2160,DaylightRoad2_3840x2160,CatRobot_3840x2160,Campfire_3840x2160',
             "A": 'Tango2_1920x1080,ParkRunning3_1920x1080,FoodMarket4_1920x1080,DaylightRoad2_1920x1080,CatRobot_1920x1080,Campfire_1920x1080',
             "B": 'RitualDance_1920x1080,ParkScene_1920x1080,MarketPlace_1920x1080,Kimono1_1920x1080,Cactus_1920x1080,BasketballDrive_1920x1080,BQTerrace_1920x1080',
             "C": 'RaceHorses_832x480,PartyScene_832x480,BasketballDrill_832x480,BQMall_832x480',
             "D": 'BlowingBubbles_416x240,BasketballPass_416x240,BQSquare_416x240,RaceHorses_416x240',
             "E": 'KristenAndSara_1280x720,Johnny_1280x720,FourPeople_1280x720'}


class Provider(object):
    def __init__(self, batch_size, num_workers, scale, datasetNum, path, path2, patch_size, color_space, quality_scale, dataReadDir):
        # Data preparing
        if color_space == 'RGB':
            self.data = DIV2K(scale, path, patch_size)
        elif color_space == 'YUV':
            self.data = Dataset_YUV(scale, datasetNum, path, path2, patch_size, quality_scale, dataReadDir)
        else:
            raise NotImplementedError

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.color_space = color_space

        self.is_cuda = True
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

        if color_space == 'YUV':
            self.quality_scale = quality_scale

    def __len__(self):
        return int(sys.maxsize)

    def build(self):
        self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                         shuffle=False, drop_last=False, pin_memory=False))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            batch = self.data_iter.next()
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
                batch[4] = batch[4].cuda()
                batch[5] = batch[5].cuda()
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
                batch[3] = batch[3].cuda()
                batch[4] = batch[4].cuda()
                batch[5] = batch[5].cuda()
            return batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]


class DIV2K(Dataset):
    def __init__(self, scale, path, patch_size, rigid_aug=True):
        super(DIV2K, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.path = path
        self.file_list = [str(i).zfill(4) for i in range(1, 801)]  # use both train and valid

        # need about 8GB shared memory "-v '--shm-size 8gb'" for docker container
        self.hr_cache = os.path.join(path, "cache_hr.npy")
        if not os.path.exists(self.hr_cache):
            self.cache_hr()
            print("HR image cache to:", self.hr_cache)
        self.hr_ims = np.load(self.hr_cache, allow_pickle=True).item()
        print("HR image cache from:", self.hr_cache)

        self.lr_cache = os.path.join(path, "cache_lr_x{}.npy".format(self.scale))
        if not os.path.exists(self.lr_cache):
            self.cache_lr()
            print("LR image cache to:", self.lr_cache)
        self.lr_ims = np.load(self.lr_cache, allow_pickle=True).item()
        print("LR image cache from:", self.lr_cache)

    def cache_lr(self):
        lr_dict = dict()
        dataLR = os.path.join(self.path, "LR", "X{}".format(self.scale))
        for f in self.file_list:
            lr_dict[f] = np.array(Image.open(os.path.join(dataLR, f + "x{}.png".format(self.scale))))
        np.save(self.lr_cache, lr_dict, allow_pickle=True)

    def cache_hr(self):
        hr_dict = dict()
        dataHR = os.path.join(self.path, "HR")
        for f in self.file_list:
            hr_dict[f] = np.array(Image.open(os.path.join(dataHR, f + ".png")))
        np.save(self.hr_cache, hr_dict, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(self.file_list)
        lb = self.hr_ims[key]
        im = self.lr_ims[key]

        shape = im.shape
        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = random.choice([0, 1, 2])

        lb = lb[i * self.scale:i * self.scale + self.sz * self.scale, j * self.scale:j * self.scale + self.sz * self.scale, c]
        im = im[i:i + self.sz, j:j + self.sz, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                im = np.fliplr(im)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                im = np.flipud(im)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            im = np.rot90(im, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)

        return im, lb

    def __len__(self):
        return int(sys.maxsize)


class SRBenchmark(Dataset):
    def __init__(self, path, scale=4):
        super(SRBenchmark, self).__init__()
        self.ims = dict()
        self.files = dict()
        _ims_all = (5 + 14 + 100 + 100 + 109) * 2

        # for dataset in ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']:
        for dataset in ['Set5']:
            folder = os.path.join(path, dataset, 'HR')
            files = os.listdir(folder)
            files.sort()
            self.files[dataset] = files

            for i in range(len(files)):
                im_hr = np.array(Image.open(os.path.join(path, dataset, 'HR', files[i])))
                im_hr = modcrop(im_hr, scale)
                if len(im_hr.shape) == 2:
                    im_hr = np.expand_dims(im_hr, axis=2)
                    im_hr = np.concatenate([im_hr, im_hr, im_hr], axis=2)

                key = dataset + '_' + files[i][:-4]
                self.ims[key] = im_hr

                im_lr = np.array(Image.open(os.path.join(path, dataset, 'LR_bicubic/X%d' % scale, files[i])))
                if len(im_lr.shape) == 2:
                    im_lr = np.expand_dims(im_lr, axis=2)
                    im_lr = np.concatenate([im_lr, im_lr, im_lr], axis=2)

                key = dataset + '_' + files[i][:-4] + 'x%d' % scale
                self.ims[key] = im_lr

                assert (im_lr.shape[0] * scale == im_hr.shape[0])
                assert (im_lr.shape[1] * scale == im_hr.shape[1])
                assert (im_lr.shape[2] == im_hr.shape[2] == 3)
        # assert (len(self.ims.keys()) == _ims_all)


class Dataset_YUV(Dataset):
    def __init__(self, scale, datasetNum, path, path2, patch_size, quality_scale, dataReadDir, rigid_aug=True):
        super(Dataset_YUV, self).__init__()
        self.scale = scale
        self.sz = patch_size
        self.rigid_aug = rigid_aug
        self.datasetNum = datasetNum
        self.dataReadDir = dataReadDir
        self.path = path
        self.path2 = path2
        self.quality_scale = quality_scale
        self.file_list_ori = dict()
        self.file_list_rec = dict()
        self.ori_ims = dict()
        self.rec_ims = dict()
        self.ori_ims_u = dict()
        self.rec_ims_u = dict()
        self.ori_ims_v = dict()
        self.rec_ims_v = dict()

        if datasetNum == 1:
            setlist = ["DIV2K_YUV420"]
        elif datasetNum == 2:
            setlist = ["DIV2K_YUV420", "BVI-DVC"]
        else:
            raise NotImplementedError

        for setName in setlist:
            if setName == "DIV2K_YUV420":
                file_path = self.path
            elif setName == "BVI-DVC":
                file_path = self.path2
            else:
                raise NotImplementedError

            self.file_list_ori[setName] = [file_name[:-4] for file_name in
                                  os.listdir(os.path.join(file_path, setName + "_ori"))]
            self.file_list_rec[setName] = [file_name[:-4] for file_name in
                                  os.listdir(os.path.join(file_path, setName + "_rec" + str(quality_scale)))]
            self.file_list_ori[setName].sort()
            self.file_list_rec[setName].sort()

            self.ori_cache_y = os.path.join(dataReadDir, setName + "_cache_ori_y.npy")
            self.ori_cache_u = os.path.join(dataReadDir, setName + "_cache_ori_u.npy")
            self.ori_cache_v = os.path.join(dataReadDir, setName + "_cache_ori_v.npy")
            if not os.path.exists(self.ori_cache_y):
                self.cache_ori(setName, file_path)
                print("Ori image of codec cache to:", self.ori_cache_y, self.ori_cache_u, self.ori_cache_v)
            self.ori_ims[setName] = np.load(self.ori_cache_y, allow_pickle=True).item()
            print("Ori image (Y) of codec cache from:", self.ori_cache_y)
            self.ori_ims_u[setName] = np.load(self.ori_cache_u, allow_pickle=True).item()
            print("Ori image (U) of codec cache from:", self.ori_cache_u)
            self.ori_ims_v[setName] = np.load(self.ori_cache_v, allow_pickle=True).item()
            print("Ori image (V) of codec cache from:", self.ori_cache_v)

            self.rec_cache_y = os.path.join(dataReadDir, setName + "_cache_rec_qp{}_y.npy".format(self.quality_scale))
            self.rec_cache_u = os.path.join(dataReadDir, setName + "_cache_rec_qp{}_u.npy".format(self.quality_scale))
            self.rec_cache_v = os.path.join(dataReadDir, setName + "_cache_rec_qp{}_v.npy".format(self.quality_scale))
            if not os.path.exists(self.rec_cache_y):
                self.cache_rec(setName, file_path)
                print("Rec image of codec cache to:", self.rec_cache_y, self.rec_cache_u, self.rec_cache_v)
            self.rec_ims[setName] = np.load(self.rec_cache_y, allow_pickle=True).item()
            print("Rec image (Y) of codec cache from:", self.rec_cache_y)
            self.rec_ims_u[setName] = np.load(self.rec_cache_u, allow_pickle=True).item()
            print("Rec image (U) of codec cache from:", self.rec_cache_u)
            self.rec_ims_v[setName] = np.load(self.rec_cache_v, allow_pickle=True).item()
            print("Rec image (V) of codec cache from:", self.rec_cache_v)

    def cache_ori(self, setName, file_path):
        ori_dict_y = dict()
        ori_dict_u = dict()
        ori_dict_v = dict()
        dataOri = os.path.join(file_path, setName + "_ori")

        for f in self.file_list_ori[setName]:
            if setName == "DIV2K_YUV420":
                yuv_size = f.split('_')[1].split('x')
                yuv_width = int(yuv_size[0])
                yuv_height = int(yuv_size[1])
            elif setName == "BVI-DVC":
                yuv_width = BVI_DVC_size[f[0]][0]
                yuv_height = BVI_DVC_size[f[0]][1]
            else:
                raise NotImplementedError
            y_bytesize = yuv_width * yuv_height
            with open(os.path.join(dataOri, f + ".yuv"), 'rb') as yuv:
                ori_dict_y[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize)
                                               .reshape(yuv_height, yuv_width), axis=2)
                ori_dict_u[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize // 4)
                                               .reshape(yuv_height // 2, yuv_width // 2), axis=2)
                ori_dict_v[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize // 4)
                                               .reshape(yuv_height // 2, yuv_width // 2), axis=2)
        np.save(self.ori_cache_y, ori_dict_y, allow_pickle=True)
        np.save(self.ori_cache_u, ori_dict_u, allow_pickle=True)
        np.save(self.ori_cache_v, ori_dict_v, allow_pickle=True)

    def cache_rec(self, setName, file_path):
        rec_dict_y = dict()
        rec_dict_u = dict()
        rec_dict_v = dict()
        dataRec = os.path.join(file_path, setName + "_rec" + str(self.quality_scale))

        for f in self.file_list_rec[setName]:
            if setName == "DIV2K_YUV420":
                yuv_size = f.split('_')[2].split('x')
                yuv_width = int(yuv_size[0])
                yuv_height = int(yuv_size[1])
            elif setName == "BVI-DVC":
                yuv_width = BVI_DVC_size[f.split('_')[1][0]][0]
                yuv_height = BVI_DVC_size[f.split('_')[1][0]][1]
            else:
                raise NotImplementedError
            y_bytesize = yuv_width * yuv_height
            with open(os.path.join(dataRec, f + ".yuv"), 'rb') as yuv:
                rec_dict_y[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize)
                                               .reshape(yuv_height, yuv_width), axis=2)
                rec_dict_u[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize // 4)
                                               .reshape(yuv_height // 2, yuv_width // 2), axis=2)
                rec_dict_v[f] = np.expand_dims(np.fromfile(yuv, dtype=np.uint8, count=y_bytesize // 4)
                                               .reshape(yuv_height // 2, yuv_width // 2), axis=2)
        np.save(self.rec_cache_y, rec_dict_y, allow_pickle=True)
        np.save(self.rec_cache_u, rec_dict_u, allow_pickle=True)
        np.save(self.rec_cache_v, rec_dict_v, allow_pickle=True)

    def __getitem__(self, _dump):
        key = random.choice(list(self.file_list_ori.keys()))
        key1 = random.choice(self.file_list_ori[key])

        if key == "BVI-DVC":
            lb = self.ori_ims[key][key1]
            lb_u = self.ori_ims_u[key][key1]
            lb_v = self.ori_ims_v[key][key1]
            key1 = key1.split("_")[0]
        else:
            lb = self.ori_ims[key][key1]
            lb_u = self.ori_ims_u[key][key1]
            lb_v = self.ori_ims_v[key][key1]
        key2 = "rec_" + key1 + "_QP{}".format(self.quality_scale)
        im = self.rec_ims[key][key2]
        im_u = self.rec_ims_u[key][key2]
        im_v = self.rec_ims_v[key][key2]

        if key == "BVI-DVC":
            shape = [im.shape[0], im.shape[1]]
            shape[0] -= BVI_DVC_size[key1.split("_")[0][0]][2] * 2
        else:
            shape = im.shape

        i = random.randint(0, shape[0] - self.sz)
        j = random.randint(0, shape[1] - self.sz)
        c = 0

        lb = lb[i:i + self.sz, j:j + self.sz, c]
        im = im[i:i + self.sz, j:j + self.sz, c]
        lb_u = lb_u[i // 2:(i + self.sz) // 2, j // 2:(j + self.sz) // 2, c]
        lb_v = lb_v[i // 2:(i + self.sz) // 2, j // 2:(j + self.sz) // 2, c]
        im_u = im_u[i // 2:(i + self.sz) // 2, j // 2:(j + self.sz) // 2, c]
        im_v = im_v[i // 2:(i + self.sz) // 2, j // 2:(j + self.sz) // 2, c]

        if self.rigid_aug:
            if random.uniform(0, 1) < 0.5:
                lb = np.fliplr(lb)
                lb_u = np.fliplr(lb_u)
                lb_v = np.fliplr(lb_v)
                im = np.fliplr(im)
                im_u = np.fliplr(im_u)
                im_v = np.fliplr(im_v)

            if random.uniform(0, 1) < 0.5:
                lb = np.flipud(lb)
                lb_u = np.flipud(lb_u)
                lb_v = np.flipud(lb_v)
                im = np.flipud(im)
                im_u = np.flipud(im_u)
                im_v = np.flipud(im_v)

            k = random.choice([0, 1, 2, 3])
            lb = np.rot90(lb, k)
            lb_u = np.rot90(lb_u, k)
            lb_v = np.rot90(lb_v, k)
            im = np.rot90(im, k)
            im_u = np.rot90(im_u, k)
            im_v = np.rot90(im_v, k)

        lb = np.expand_dims(lb.astype(np.float32) / 255.0, axis=0)
        im = np.expand_dims(im.astype(np.float32) / 255.0, axis=0)
        lb_u = np.expand_dims(lb_u.astype(np.float32) / 255.0, axis=0)
        im_u = np.expand_dims(im_u.astype(np.float32) / 255.0, axis=0)
        lb_v = np.expand_dims(lb_v.astype(np.float32) / 255.0, axis=0)
        im_v = np.expand_dims(im_v.astype(np.float32) / 255.0, axis=0)

        return im, lb, im_u, lb_u, im_v, lb_v

    def __len__(self):
        return int(sys.maxsize)


class Benchmark(Dataset):
    def __init__(self, path, valid_per_size, valid_skip_size, qualityScale, validFast, scale=4):
        super(Benchmark, self).__init__()
        self.ims_ori = dict()
        self.ims_rec = dict()
        self.files_ori = dict()
        self.files_rec = dict()
        self.validPerSize = valid_per_size
        self.validSkipSize = valid_skip_size

        for dataset in ['VVC_AI']:
            if qualityScale != 37:
                folder_ori, folder_rec = os.path.join(path, dataset, 'ori'), os.path.join(path, dataset, 'rec' + str(qualityScale))
            else:
                folder_ori, folder_rec = os.path.join(path, dataset, 'ori'), os.path.join(path, dataset, 'rec')
            files_ori = [file_name[:-4] for file_name in os.listdir(folder_ori)]
            files_rec = [file_name[:-4] for file_name in os.listdir(folder_rec)]
            if validFast:
                for class_id in ["Ah", "A", "B", "C", "E"]:
                    for delete_seq in VVC_class[class_id].split(','):
                        files_rec = [file for file in files_rec if delete_seq not in file]
                        files_ori = [file for file in files_ori if delete_seq not in file]
            files_ori.sort()
            files_rec.sort()
            print("Valid-ori:", files_ori)
            print("Valid-rec:", files_rec)
            self.files_ori[dataset], self.files_rec[dataset] = files_ori, files_rec

            for i in range(len(files_ori)):
                im_ori = dict()
                im_rec = dict()
                yuv_size = files_ori[i].split('_')[1].split('x')
                width, height = int(yuv_size[0]), int(yuv_size[1])
                frame_bytesize = width * height * 3 // 2

                with open(os.path.join(folder_ori, files_ori[i] + ".yuv"), 'rb') as yuv:
                    for frame in range(valid_per_size):
                        if frame % valid_skip_size == 0:
                            video_data = np.frombuffer(yuv.read(frame_bytesize), dtype=np.uint8)
                            im_ori_y = modcrop(np.expand_dims(video_data[:width * height]
                                                          .reshape((height, width)), axis=2), scale)
                            im_ori_u = modcrop(np.expand_dims(video_data[width * height:(width * height * 5) // 4]
                                                          .reshape((height // 2, width // 2)), axis=2), scale)
                            im_ori_v = modcrop(np.expand_dims(video_data[(width * height * 5) // 4:]
                                                          .reshape((height // 2, width // 2)), axis=2), scale)
                            im_ori[frame] = {"y": im_ori_y, "u": im_ori_u, "v": im_ori_v}
                        else:
                            yuv.seek(frame_bytesize, os.SEEK_CUR)
                key = dataset + '_' + files_ori[i]
                self.ims_ori[key] = im_ori

                with open(os.path.join(folder_rec, files_rec[i] + ".yuv"), 'rb') as yuv:
                    for frame in range(valid_per_size):
                        if frame % valid_skip_size == 0:
                            video_data = np.frombuffer(yuv.read(frame_bytesize), dtype=np.uint8)
                            im_rec_y = modcrop(np.expand_dims(video_data[:width * height]
                                                          .reshape((height, width)), axis=2), scale)
                            im_rec_u = modcrop(np.expand_dims(video_data[width * height:(width * height * 5) // 4]
                                                          .reshape((height // 2, width // 2)), axis=2), scale)
                            im_rec_v = modcrop(np.expand_dims(video_data[(width * height * 5) // 4:]
                                                          .reshape((height // 2, width // 2)), axis=2), scale)
                            im_rec[frame] = {"y": im_rec_y, "u": im_rec_u, "v": im_rec_v}
                        else:
                            yuv.seek(frame_bytesize, os.SEEK_CUR)
                key = dataset + '_' + files_rec[i]
                self.ims_rec[key] = im_rec

                assert (im_ori_y.shape[0] == im_rec_y.shape[0])
                assert (im_ori_y.shape[1] == im_ori_y.shape[1])
                assert (im_ori_y.shape[2] == im_ori_y.shape[2])
                assert (len(files_ori) == len(files_rec))
                assert (len(self.ims_ori) == len(self.ims_rec))