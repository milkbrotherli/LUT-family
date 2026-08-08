import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import model
from data_uv import Provider, SRBenchmark, Benchmark

sys.path.insert(0, "../")
from common.option import TrainOptions
from common.utils import PSNR, logger_info, _rgb2ycbcr

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

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

close_grad = ["alphas"]

def round_func(input):
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data

    return out

def SaveCheckpoint(model_G, opt_G, opt, scheduler, i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G.state_dict(), os.path.join(opt.expDir, 'Model_{:06d}{}.pth'.format(i, str_best)))
    torch.save(opt_G.state_dict(), os.path.join(opt.expDir, 'Opt_{:06d}{}.pth'.format(i, str_best)))
    torch.save(scheduler.state_dict(), os.path.join(opt.expDir, 'Schedule_{:06d}{}.pth'.format(i, str_best)))
    logger.info("Checkpoint saved {}".format(str(i)))

def valid_steps(model_G, valid, opt, iter):
    if opt.debug:
        datasets = ['Set5']
    else:
        datasets = ['Set5']  # , 'Set14', 'B100', 'Urban100', 'Manga109'

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            files = valid.files[datasets[i]]
            result_path = os.path.join(opt.valoutDir, datasets[i])
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = valid.ims[key]
                input_im = valid.ims[key + 'x%d' % opt.scale]

                input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()

                pred = model_G(im, 'valid', opt)

                pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, opt.scale))

                if iter < 10000:
                    input_img = np.round(np.clip(input_im * 255.0, 0, 255)).astype(np.uint8)
                    Image.fromarray(input_img).save(os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                    Image.fromarray(lb.astype(np.uint8)).save(os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))
                Image.fromarray(pred).save(os.path.join(result_path, '{}_net.png'.format(key.split('_')[-1])))

            logger.info('Iter {} | Dataset {} | AVG Val PSNR: {:02f}'.format(iter, datasets[i], np.mean(np.asarray(psnrs))))
            writer.add_scalar('PSNR_valid/{}'.format(datasets[i]), np.mean(np.asarray(psnrs)), iter)
            writer.flush()

def valid_steps_postprocessing(model_G, valid, opt, iter):
    if opt.debug:
        datasets = ['VVC_AI']
    else:
        datasets = ['VVC_AI']

    logger.info("Valid Steps is Starting!")

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnr_per_seq_y = dict()
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
                        lb = (lbs[frame])['y']

                        input_im = (input_ims[frame])['y'].astype(np.float32) / 255.0
                        im = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                        pred = model_G(im, 'valid')
                        pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                        pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                        left_y, right_y = pred[:, :, 0], lb[:, :, 0]
                        ims_psnr.append(PSNR(left_y, right_y, opt.scale))

                        # temp_u_lb, temp_v_lb = (lbs[frame])['u'], (lbs[frame])['v']
                        # temp_u_rec, temp_v_rec = (input_ims[frame])['u'], (input_ims[frame])['v']

                        if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                            orig_rec_im = (input_ims[frame])['y']
                            rec_y = orig_rec_im[:, :, 0]
                            ori_y = lb[:, :, 0]
                            recs_psnr.append(PSNR(rec_y, ori_y, opt.scale))
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
                        # with open(os.path.join(output_path, 'val_{}_iter{}.yuv'
                        #         .format(str(frame), str(iter))), "wb") as file:
                        #     file.write(pred.tobytes())
                        #     file.write(temp_u_rec.tobytes())
                        #     file.write(temp_v_rec.tobytes())
                    else:
                        continue

                if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                    psnr_orig_rec[files_ori[j]] = np.mean(np.asarray(recs_psnr))
                psnr_per_seq_y[files_ori[j]] = np.mean(np.asarray(ims_psnr))

                # class division
                if opt.valA:
                    for Class in ['A', 'B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_y[files_ori[j]]
                            break
                else:
                    for Class in ['B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_y[files_ori[j]]
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
                                            np.mean(np.asarray(list(psnr_per_seq_y.values())))),
                                            np.mean(np.asarray(list(psnr_per_seq_y.values())))-
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
                            np.mean(np.asarray(list(psnr_per_seq_y.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_y.values()))) -
                            np.mean(np.asarray(list(psnr_orig_rec.values())))))

            writer.add_scalar('PSNR_valid/{}'.format(datasets[i]), np.mean(np.asarray(list(psnr_per_seq_y.values()))), iter)
            writer.flush()

def valid_steps_postprocessing_v(model_G, valid, opt, iter):
    if opt.debug:
        datasets = ['VVC_AI']
    else:
        datasets = ['VVC_AI']

    logger.info("Valid Steps is Starting!")

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnr_per_seq_v = dict()
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
                        lb = (lbs[frame])['v']
                        input_im = (input_ims[frame])['y'].astype(np.float32) / 255.0
                        input_im_u = (input_ims[frame])['u'].astype(np.float32) / 255.0
                        input_im_v = (input_ims[frame])['v'].astype(np.float32) / 255.0

                        im = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                        im_u = torch.Tensor(np.expand_dims(np.transpose(input_im_u, [2, 0, 1]), axis=0)).cuda()
                        im_v = torch.Tensor(np.expand_dims(np.transpose(input_im_v, [2, 0, 1]), axis=0)).cuda()
                        # Reuse cc_u_alf with V in the target-color position.
                        pred = model_G(im, im_v, im_u, 'valid')
                        pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                        pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                        left_v, right_v = pred[:, :, 0], lb[:, :, 0]
                        ims_psnr.append(PSNR(left_v, right_v, opt.scale))

                        # temp_u_lb, temp_v_lb = (lbs[frame])['u'], (lbs[frame])['v']
                        temp_y_rec, temp_u_rec = (input_ims[frame])['y'], (input_ims[frame])['u']

                        if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                            orig_rec_im = (input_ims[frame])['v']
                            rec_v = orig_rec_im[:, :, 0]
                            ori_v = lb[:, :, 0]
                            recs_psnr.append(PSNR(rec_v, ori_v, opt.scale))
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
                        with open(os.path.join(output_path, 'val_{}.yuv'
                                .format(str(frame))), "wb") as file:
                            file.write(temp_y_rec.tobytes())
                            file.write(temp_u_rec.tobytes())
                            file.write(pred.tobytes())
                    else:
                        continue

                if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                    psnr_orig_rec[files_ori[j]] = np.mean(np.asarray(recs_psnr))
                psnr_per_seq_v[files_ori[j]] = np.mean(np.asarray(ims_psnr))

                # class division
                if opt.valA:
                    for Class in ['A', 'B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_v[files_ori[j]]
                            break
                else:
                    for Class in ['B', 'C', 'D', 'E']:
                        if files_ori[j] in CTC_class[Class]:
                            if iter <= (opt.startIter + opt.valStep) or iter <= (opt.valStartIter + opt.valStep):
                                (psnr_orig_rec_per_class[Class])[files_ori[j]] = psnr_orig_rec[files_ori[j]]
                            (psnr_per_class[Class])[files_ori[j]] = psnr_per_seq_v[files_ori[j]]
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
                                            np.mean(np.asarray(list(psnr_per_seq_v.values())))),
                                            np.mean(np.asarray(list(psnr_per_seq_v.values())))-
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
                            np.mean(np.asarray(list(psnr_per_seq_v.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_v.values()))) -
                            np.mean(np.asarray(list(psnr_orig_rec.values())))))

            writer.add_scalar('PSNR_valid/{}'.format(datasets[i]), np.mean(np.asarray(list(psnr_per_seq_v.values()))), iter)
            writer.flush()

if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    if os.path.normpath(opt.expDir) == os.path.normpath("../runs/experiment"):
        opt.expDir = "../checkpoints/v"
    if os.path.normpath(opt.loadDir) == os.path.normpath("../checkpoints"):
        opt.loadDir = "../checkpoints/v"
    opt.valoutDir = os.path.join(opt.expDir, "val")
    os.makedirs(opt.valoutDir, exist_ok=True)
    opt_inst.save_options(opt)

    writer = SummaryWriter(log_dir=opt.logDir)

    logger_name = 'train'
    logger_dir = os.path.join(opt.expDir, logger_name + '.txt')
    logger_info(logger_name, logger_dir)
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    stage1_modes = [i for i in opt.stage1_modes]
    stage2_modes = [i for i in opt.stage2_modes]
    stage3_modes = [i for i in opt.stage3_modes]
    stage4_modes = [i for i in opt.stage4_modes]

    stages = opt.stages

    model = model.LUT_ILF_Net_cc_u_alf

    if opt.weight:
        model_G = model(nf=opt.nf, scale=opt.scale, stage1_modes=opt.stage1_modes, weight=opt.weight, ps_error=opt.ps_error).cuda()
    else:
        model_G = model(nf=opt.nf, scale=opt.scale, stage1_modes=opt.stage1_modes, ps_error=opt.ps_error).cuda()

    if opt.gpuNum > 1:
        model_G = torch.nn.DataParallel(model_G, device_ids=list(range(opt.gpuNum)))

    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))

    if opt.steplr:
        opt_G = optim.Adam(params_G, lr=opt.lr0)
        scheduler = optim.lr_scheduler.StepLR(opt_G, step_size=opt.step, gamma=opt.reduce)
    else:
        opt_G = optim.Adam(params_G, lr=opt.lr0, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weightDecay, amsgrad=False)
        if opt.lr1 < 0:
            lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * 0.8 + 0.2
        else:
            lr_b = opt.lr1 / opt.lr0
            lr_a = 1 - lr_b
            lf = lambda x: (((1 + math.cos(x * math.pi / opt.totalIter)) / 2) ** 1.0) * lr_a + lr_b
        scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    if opt.startIter > 0:
        lm = torch.load(os.path.join(opt.loadDir, 'Model_{:06d}.pth'.format(opt.startIter)))
        if opt.gpuNum > 1:
            new_state_dict = {f'module.{k}': v for k, v in lm.items()}
            model_G.load_state_dict(new_state_dict, strict=True)
        else:
            model_G.load_state_dict(lm, strict=True)

        # load optimization
        lm = torch.load(os.path.join(opt.loadDir, 'Opt_{:06d}.pth'.format(opt.startIter)))
        opt_G.load_state_dict(lm)
        logger.info("Model Loading: Model: Model_{:06d}.pth, Opt: Opt_{:06d}.pth".format(opt.startIter, opt.startIter))

        # lr scheduler
        if opt.schedule:
            sch_path = os.path.join(opt.loadDir, 'Schedule_{:06d}.pth'.format(opt.startIter))
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
    if opt.startIter > 0:
        if opt.gpuNum > 1:
            if opt.task == 'in-loop-filter':
                valid_steps_postprocessing_v(model_G.module, valid, opt, opt.startIter)
            else:
                valid_steps(model_G.module, valid, opt, opt.startIter)
        else:
            if opt.task == 'in-loop-filter':
                valid_steps_postprocessing_v(model_G, valid, opt, opt.startIter)
            else:
                valid_steps(model_G, valid, opt, opt.startIter)

    if opt.validation:
        exit()

    i = opt.startIter

    for i in range(opt.startIter + 1, opt.totalIter + 1):
        model_G.train()

        st = time.time()
        im, lb, im_u, lb_u, im_v, lb_v = train_iter.next()
        im = im.cuda()
        im_u = im_u.cuda()
        im_v = im_v.cuda()
        lb_v = lb_v.cuda()

        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()

        pred_v = model_G(im, im_v, im_u, 'train')

        loss_G = F.mse_loss(pred_v, lb_v)
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        accum_samples += opt.batchSize
        l_accum[0] += loss_G.item()

        if i % opt.displayStep == 0:
            writer.add_scalar('loss_Pixel', l_accum[0] / opt.displayStep, i)
            logger.info("{} | Iter:{:6d}, lr:{:6f}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                opt.expDir, i, opt_G.param_groups[0]['lr'], accum_samples, l_accum[0] / opt.displayStep, dT / opt.displayStep,
                                              rT / opt.displayStep))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        if i % opt.saveStep == 0:
            if opt.gpuNum > 1:
                SaveCheckpoint(model_G.module, opt_G, opt, scheduler, i)
            else:
                SaveCheckpoint(model_G, opt_G, opt, scheduler, i)

            if opt.weight:
                if opt.gpuNum > 1:
                    probs1 = model_G.module.convblock1.get_probs()
                    probs2 = model_G.module.convblock2.get_probs()
                    probs3 = model_G.module.convblock3.get_probs()
                    probs4 = model_G.module.upblock.get_probs()
                    logger.info('Stage1 Pattern Weight: {}, {}'.format(probs1[0].detach().cpu().numpy(), probs1[1].detach().cpu().numpy()))
                    logger.info('Stage2 Pattern Weight: {}, {}'.format(probs2[0].detach().cpu().numpy(), probs2[1].detach().cpu().numpy()))
                    logger.info('Stage3 Pattern Weight: {}'.format(probs3[0].detach().cpu().numpy()))
                    logger.info('Stage4_1 Pattern Weight: {}'.format(probs4[0].detach().cpu().numpy()))
                    logger.info('Stage4_2 Pattern Weight: {}'.format(probs4[1].detach().cpu().numpy()))
                    logger.info('Stage4_3 Pattern Weight: {}'.format(probs4[2].detach().cpu().numpy()))
                else:
                    probs1 = model_G.convblock1.get_probs()
                    probs2 = model_G.convblock2.get_probs()
                    probs3 = model_G.convblock3.get_probs()
                    probs4 = model_G.upblock.get_probs()
                    logger.info('Stage1 Pattern Weight: {}, {}'.format(probs1[0].detach().cpu().numpy(), probs1[1].detach().cpu().numpy()))
                    logger.info('Stage2 Pattern Weight: {}, {}'.format(probs2[0].detach().cpu().numpy(), probs2[1].detach().cpu().numpy()))
                    logger.info('Stage3 Pattern Weight: {}'.format(probs3[0].detach().cpu().numpy()))
                    logger.info('Stage4_1 Pattern Weight: {}'.format(probs4[0].detach().cpu().numpy()))
                    logger.info('Stage4_2 Pattern Weight: {}'.format(probs4[1].detach().cpu().numpy()))
                    logger.info('Stage4_3 Pattern Weight: {}'.format(probs4[2].detach().cpu().numpy()))

        if i % opt.saveStep == 0 and i % opt.valStep == 0 and i >= opt.valStartIter:
            if opt.gpuNum > 1:
                if opt.task == 'in-loop-filter':
                    valid_steps_postprocessing_v(model_G.module, valid, opt, i)
                else:
                    valid_steps(model_G.module, valid, opt, i)
            else:
                if opt.task == 'in-loop-filter':
                    valid_steps_postprocessing_v(model_G, valid, opt, i)
                else:
                    valid_steps(model_G, valid, opt, i)

    logger.info("Complete")
