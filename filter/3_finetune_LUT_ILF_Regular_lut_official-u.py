import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import model as Model
from data_uv import Provider, SRBenchmark, Benchmark

sys.path.insert(0, "../")
from common.option import TrainOptions
from common.utils import PSNR, logger_info

torch.backends.cudnn.benchmark = True

psnr_orig_rec = dict()
psnr_orig_rec_per_class = dict()
psnr_orig_rec_per_class["A"] = dict()
psnr_orig_rec_per_class["B"], psnr_orig_rec_per_class["C"] = dict(), dict()
psnr_orig_rec_per_class["D"], psnr_orig_rec_per_class["E"] = dict(), dict()

CTC_class = {"A": 'Tango2_1920x1080, ParkRunning3_1920x1080, FoodMarket4_1920x1080, DaylightRoad2_1920x1080, '
                  'CatRobot_1920x1080, Campfire_1920x1080',
             "B": 'RitualDance_1920x1080, ParkScene_1920x1080, MarketPlace_1920x1080, Kimono1_1920x1080, '
                  'Cactus_1920x1080, BasketballDrive_1920x1080, BQTerrace_1920x1080',
             "C": 'RaceHorses_832x480, PartyScene_832x480, BasketballDrill_832x480, BQMall_832x480',
             "D": 'BlowingBubbles_416x240, BasketballPass_416x240, BQSquare_416x240, RaceHorses_416x240',
             "E": 'KristenAndSara_1280x720, Johnny_1280x720, FourPeople_1280x720'}


def valid_steps_postprocessing_u(model_G, valid, opt, iter):
    if opt.debug:
        datasets = ['VVC_AI']
    else:
        datasets = ['VVC_AI']

    logger.info("Valid Steps is Starting!")

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnr_per_seq_u = dict()
            psnr_per_class = dict()
            if opt.valA:
                psnr_per_class['A'] = dict()
            psnr_per_class['B'], psnr_per_class['C'] = dict(), dict()
            psnr_per_class['D'], psnr_per_class['E'] = dict(), dict()

            files_ori = valid.files_ori[datasets[i]]
            files_rec = valid.files_rec[datasets[i]]
            result_path = os.path.join(opt.valoutDir, datasets[i])
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files_ori)):
                key_ori = datasets[i] + '_' + files_ori[j]
                key_rec = datasets[i] + '_' + files_rec[j]
                lbs = valid.ims_ori[key_ori]
                input_ims = valid.ims_rec[key_rec]
                logger.info("Valid Pair: ori ({}), rec ({})".format(key_ori, key_rec))

                if files_ori[j] in ['RaceHorses_832x480']:
                    output_path = os.path.join(result_path, files_ori[j].split('_')[0] + 'C_QP' + str(opt.qualityScale))
                else:
                    output_path = os.path.join(result_path, files_ori[j].split('_')[0] + '_QP' + str(opt.qualityScale))
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                ims_psnr = []
                recs_psnr =[]

                for frame in range(opt.validPerSize):
                    if frame % opt.validSkipSize == 0:
                        lb = (lbs[frame])['u']
                        input_im = (input_ims[frame])['y'].astype(np.float32) / 255.0
                        input_im_u = (input_ims[frame])['u'].astype(np.float32) / 255.0
                        input_im_v = (input_ims[frame])['v'].astype(np.float32) / 255.0

                        im = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                        im_u = torch.Tensor(np.expand_dims(np.transpose(input_im_u, [2, 0, 1]), axis=0)).cuda()
                        im_v = torch.Tensor(np.expand_dims(np.transpose(input_im_v, [2, 0, 1]), axis=0)).cuda()
                        pred = model_G(im, im_u, im_v, 'valid')
                        pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                        pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                        left_u, right_u = pred[:, :, 0], lb[:, :, 0]
                        ims_psnr.append(PSNR(left_u, right_u, opt.scale))
                        temp_y_rec, temp_v_rec = (input_ims[frame])['y'], (input_ims[frame])['v']
                        # temp_u_lb, temp_v_lb = (lbs[frame])['u'], (lbs[frame])['v']
                        # temp_u_rec, temp_v_rec = (input_ims[frame])['u'], (input_ims[frame])['v']

                        if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                            orig_rec_im = (input_ims[frame])['u']
                            rec_u = orig_rec_im[:, :, 0]
                            ori_u = lb[:, :, 0]
                            recs_psnr.append(PSNR(rec_u, ori_u, opt.scale))
                            # with open(os.path.join(output_path, files_ori[j].split('_')[0] + '_{}_orig.yuv'
                            #         .format(str(frame))),  "wb") as file:
                            #     file.write(lb.tobytes())
                            #     file.write(temp_u_lb.tobytes())
                            #     file.write(temp_v_lb.tobytes())
                            # with open(os.path.join(output_path, files_ori[j].split('_')[0] + '_{}_rec.yuv'
                            #         .format(str(frame))),  "wb") as file:
                            #     file.write(orig_rec_im.tobytes())
                            #     file.write(temp_u_rec.tobytes())
                            #     file.write(temp_v_rec.tobytes())
                        with open(os.path.join(output_path, 'val_{}.yuv'.format(str(frame))), "wb") as file:
                            file.write(temp_y_rec.tobytes())
                            file.write(pred.tobytes())
                            file.write(temp_v_rec.tobytes())
                    else:
                        continue

                if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                    psnr_orig_rec[files_ori[j]] = np.mean(np.asarray(recs_psnr))
                psnr_per_seq_u[files_ori[j]] = np.mean(np.asarray(ims_psnr))

                # class division
                if opt.valA:
                    for Class in ['A', 'B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_u[files_ori[j]]
                            break
                else:
                    for Class in ['B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_u[files_ori[j]]
                            break
            if opt.valA:
                for Class in ['A', 'B', 'C', 'D', 'E']:
                    if len(psnr_per_class[Class]):
                        logger.info('Iter {} | Each Class {} | Rec PSNR: {:02f}  AVG Val PSNR: {:02f}'.format(iter,
                                            Class, np.mean(np.asarray(list(psnr_orig_rec_per_class[Class].values()))),
                                            np.mean(np.asarray(list(psnr_per_class[Class].values())))))
                        for key in psnr_orig_rec_per_class[Class].keys():
                            logger.info('Iter {} | Class {}: {} | Rec PSNR: {:02f} Val PSNR: {:02f}'.format(iter,
                                                Class, key, (psnr_orig_rec_per_class[Class])[key],
                                                (psnr_per_class[Class])[key]))
                logger.info('Iter {} | Dataset {} | Rec PSNR: {:02f}  AVG Val PSNR: {:02f}  Difference: {:02f}'
                                            .format(iter, datasets[i], np.mean(np.asarray(list(psnr_orig_rec.values()))),
                                            np.mean(np.asarray(list(psnr_per_seq_u.values())))),
                                            np.mean(np.asarray(list(psnr_per_seq_u.values())))-
                                            np.mean(np.asarray(list(psnr_orig_rec.values()))))

            else:
                for Class in ['B', 'C', 'D', 'E']:
                    if len(psnr_per_class[Class]):
                        logger.info('Iter {} | Each Class {} | Rec PSNR: {:02f}  AVG Val PSNR: {:02f}'.format(iter,
                                                Class, np.mean(np.asarray(list(psnr_orig_rec_per_class[Class].values()))),
                                                np.mean(np.asarray(list(psnr_per_class[Class].values())))))
                        for key in psnr_orig_rec_per_class[Class].keys():
                            logger.info('Iter {} | Class {}: {} | Rec PSNR: {:02f} Val PSNR: {:02f}'.format(iter,
                                                Class, key, (psnr_orig_rec_per_class[Class])[key],
                                                (psnr_per_class[Class])[key]))
                logger.info('Iter {} | Dataset {} | Rec PSNR: {:02f}  AVG Val PSNR: {:02f}  Difference: {:02f}'
                            .format(iter, datasets[i], np.mean(np.asarray(list(psnr_orig_rec.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_u.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_u.values()))) -
                            np.mean(np.asarray(list(psnr_orig_rec.values())))))


if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    if os.path.normpath(opt.expDir) in {
        os.path.normpath("../runs/experiment"),
        os.path.normpath("../runs/finetune"),
    }:
        opt.expDir = "../runs/finetune/u"
    if os.path.normpath(opt.lutDir) == os.path.normpath("../luts/regular"):
        opt.lutDir = "../luts/regular/u"
    if os.path.normpath(opt.lutSaveDir) == os.path.normpath("../luts/finetuned"):
        opt.lutSaveDir = "../luts/finetuned/u"
    opt.valoutDir = os.path.join(opt.expDir, "val")
    for directory in (opt.expDir, opt.lutDir, opt.lutSaveDir, opt.valoutDir):
        os.makedirs(directory, exist_ok=True)

    logger_name = 'lut-regular-u-alf-ft'
    logger_dir = os.path.join(opt.expDir, logger_name + '.txt')
    logger_info(logger_name, logger_dir)
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    stage1_modes = [i for i in opt.stage1_modes]
    stage2_modes = [i for i in opt.stage2_modes]
    stage3_modes = [i for i in opt.stage3_modes]
    stage4_modes = [i for i in opt.stage4_modes]

    stages = opt.stages

    model = Model.LUT_ILF_Net_cc_u_alf_LUT

    model_G = model(lut_folder=opt.lutDir, stages=stages, modes=stage1_modes, lutName=opt.lutName, upscale=opt.scale, interval=opt.interval, conversion=opt.conversion).cuda()

    if opt.gpuNum > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=list(range(opt.gpuNum)))

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)

    # Learning rate schedule
    if opt.lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.maxIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = opt.lr1 / opt.lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / opt.maxIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Load saved params
    if opt.startIter > 0:
        parameter_model = model_G.module if opt.gpuNum > 1 else model_G
        for name, parameter in parameter_model.named_parameters():
            lut_path = os.path.join(
                opt.lutSaveDir, '{}_{:06d}.npy'.format(name, opt.startIter)
            )
            lut_arr = np.load(lut_path).astype(np.float32) / 127.0
            if tuple(lut_arr.shape) != tuple(parameter.shape):
                raise ValueError(
                    'LUT shape mismatch for {}: file {}, model {}'.format(
                        name, lut_arr.shape, tuple(parameter.shape)
                    )
                )
            parameter.data.copy_(torch.from_numpy(lut_arr).to(parameter.device))

        lm = torch.load(os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter)))
        opt_G.load_state_dict(lm)
        logger.info("Model Loading: Opt: Opt_{:06d}.pth".format(opt.startIter))

        if opt.schedule:
            sch_path = os.path.join(opt.expDir, 'Schedule_{:06d}.pth'.format(opt.startIter))
            if os.path.exists(sch_path):
                lm = torch.load(sch_path)
                scheduler.load_state_dict(lm)
                logger.info("Scheduler: Sch_{:06d}.pth".format(opt.startIter))
            else:
                raise NotImplementedError
        else:
            logger.info("Without Loading Scheduler")

    # Training dataset
    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.datasetNum, opt.trainDir, opt.trainDir2,
                          opt.cropSize, opt.colorSpace, opt.qualityScale, opt.dataReadDir)

    # Valid dataset
    if opt.colorSpace == 'RGB':
        valid = SRBenchmark(opt.valDir, scale=opt.scale)
    elif opt.colorSpace == 'YUV':
        valid = Benchmark(opt.valDir, opt.validPerSize, opt.validSkipSize, opt.qualityScale, opt.validFast, scale=opt.scale)

    # Training
    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # load init validation
    if opt.startIter == 0:
        if opt.gpuNum > 1:
            valid_steps_postprocessing_u(model_G.module, valid, opt, opt.startIter)
        else:
            valid_steps_postprocessing_u(model_G, valid, opt, opt.startIter)

    i = opt.startIter

    for i in range(opt.startIter + 1, opt.maxIter + 1):
        model_G.train()

        # Data preparing
        st = time.time()
        im, lb, im_u, lb_u, im_v, lb_v = train_iter.next()
        im = im.cuda()
        im_u = im_u.cuda()
        im_v = im_v.cuda()
        lb_u = lb_u.cuda()

        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()

        pred_u = model_G(im, im_u, im_v, 'train')

        loss_G = F.mse_loss(pred_u, lb_u)
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        # Show information
        if i % opt.displayStep == 0:
            logger.info("{} | Iter:{:6d}, lr:{:6f}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                opt.expDir, i, opt_G.param_groups[0]['lr'], accum_samples, l_accum[0] / opt.displayStep, dT / opt.displayStep,
                                              rT / opt.displayStep))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        if i % opt.valStep == 0 and i > opt.valStartIter:
            if opt.gpuNum > 1:
                valid_steps_postprocessing_u(model_G.module, valid, opt, i)
            else:
                valid_steps_postprocessing_u(model_G, valid, opt, i)

        if i % opt.saveStep == 0:
            os.makedirs(opt.lutSaveDir, exist_ok=True)
            torch.save(opt_G.state_dict(), os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(i)))
            torch.save(scheduler.state_dict(), os.path.join(opt.expDir, 'Schedule_{:06d}.pth'.format(i)))
            for k, v in model_G.named_parameters():
                parameter_name = k[7:] if k.startswith('module.') else k
                ft_lut_path = os.path.join(
                    opt.lutSaveDir, "{}_{:06d}.npy".format(parameter_name, i)
                )
                lut_weight = np.round(np.clip(v.cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
                np.save(ft_lut_path, lut_weight)
                # Also maintain iteration-free files so the regular ALF LUT
                # class can test the latest checkpoint by changing lutDir only.
                latest_lut_path = os.path.join(opt.lutSaveDir, "{}.npy".format(parameter_name))
                np.save(latest_lut_path, lut_weight)
            logger.info("Checkpoint saved {}".format(str(i)))

    logger.info("Finetuned LUT saved to {}".format(opt.lutSaveDir))
    logger.info("Complete")
