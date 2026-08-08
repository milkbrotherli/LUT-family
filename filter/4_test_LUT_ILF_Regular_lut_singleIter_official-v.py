import logging
import os
import sys
import numpy as np
import torch


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
                        # cc_u_alf integrates its second argument; pass V there.
                        pred = model_G(im, im_v, im_u, 'valid')
                        pred = np.transpose(np.squeeze(pred.data.cpu().numpy(), 0), [1, 2, 0])
                        pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                        left_v, right_v = pred[:, :, 0], lb[:, :, 0]
                        ims_psnr.append(PSNR(left_v, right_v, opt.scale))

                        temp_u_lb, temp_v_lb = (lbs[frame])['u'], (lbs[frame])['v']
                        temp_y_rec, temp_u_rec, temp_v_rec = (input_ims[frame])['y'], (input_ims[frame])['u'], (input_ims[frame])['v']

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
                psnr_Difference = np.mean(np.asarray(list(psnr_per_seq_v.values()))) - np.mean(
                    np.asarray(list(psnr_orig_rec.values())))
                logger.info('Iter {} | Dataset {} | Rec PSNR: {:02f}  AVG Val PSNR: {:02f}  Difference: {:02f}'
                            .format(iter, datasets[i], np.mean(np.asarray(list(psnr_orig_rec.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_v.values()))),
                            np.mean(np.asarray(list(psnr_per_seq_v.values()))) -
                            np.mean(np.asarray(list(psnr_orig_rec.values())))))

    return psnr_Difference

if __name__ == "__main__":
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    if os.path.normpath(opt.expDir) == os.path.normpath("../runs/experiment"):
        opt.expDir = "../runs/test/v"
    if os.path.normpath(opt.lutDir) in {
        os.path.normpath("../luts/regular"),
        os.path.normpath("../luts/finetuned"),
    }:
        opt.lutDir = "../luts/finetuned/v"
    opt.valoutDir = os.path.join(opt.expDir, "test")
    os.makedirs(opt.valoutDir, exist_ok=True)

    logger_name = 'lut-regular-v-test'
    logger_dir = os.path.join(opt.expDir, logger_name + '.txt')
    logger_info(logger_name, logger_dir)
    logger = logging.getLogger(logger_name)
    logger.info(opt_inst.print_options(opt))

    stage1_modes = [i for i in opt.stage1_modes]

    stages = opt.stages

    model = Model.LUT_ILF_Net_cc_u_alf_LUT
    valid = Benchmark(opt.valDir, opt.validPerSize, opt.validSkipSize, opt.qualityScale, opt.validFast, scale=opt.scale)

    model_G = model(lut_folder=opt.lutDir, stages=stages, modes=stage1_modes, lutName=opt.lutName, upscale=opt.scale, interval=opt.interval, conversion=opt.conversion).cuda()
    iter = opt.loadIter
    PSNR_Difference = valid_steps_postprocessing_v(model_G, valid, opt, iter)
    print("PSNR_Performance:{}".format(PSNR_Difference))

    logger.info("Complete")
