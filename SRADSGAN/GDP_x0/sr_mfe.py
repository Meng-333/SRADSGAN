import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_ddpm_train_64_256.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    global scale
    if opt['datasets']['train']['l_resolution'] == 108:
        scale = 2
    elif opt['datasets']['train']['l_resolution'] == 72:
        scale = 3
    elif opt['datasets']['train']['l_resolution'] == 54:
        scale = 4
    elif opt['datasets']['train']['l_resolution'] == 27:
        scale = 8
    elif opt['datasets']['train']['l_resolution'] == 24:
        scale = 9

    # print('sr_mfe.py---scale:', scale)


    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    bic_mse = 0.0
                    bic_psnr = 0.0
                    bic_ssim = 0.0
                    bic_ergas = 0.0
                    bic_lpips = 0.0
                    avg_mse = 0.0
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_ergas = 0.0
                    avg_lpips = 0.0
                    idx = 0

                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        # if opt['model']['which_model_G'] == 'srcdm':
                        #     diffusion.srcdm_test(continous=False)
                        # else:
                        #     diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8  the bicubic img

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.tif'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.tif'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.tif'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.tif'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                            idx)

                        bmse = compare_mse(fake_img, hr_img)
                        bpsnr = compare_psnr(fake_img, hr_img)
                        bssim = compare_ssim(fake_img, hr_img, multichannel=True)
                        bergas = Metrics.calculate_ergas(fake_img, hr_img, scale=scale)
                        blpips = Metrics.calculate_lpips(fake_img, hr_img)
                        smse = compare_mse(sr_img, hr_img)
                        spsnr = compare_psnr(sr_img, hr_img)
                        sssim = compare_ssim(sr_img, hr_img, multichannel=True)
                        sergas = Metrics.calculate_ergas(sr_img, hr_img, scale=scale)
                        slpips = Metrics.calculate_lpips(sr_img, hr_img)
                        #result_imgs = [hr_img.cpu(), lr_img.cpu(), fake_img.cpu(), sr_img.cpu()]
                        result_imgs = [hr_img, lr_img, fake_img, sr_img]
                        mses = [None, None, bmse, smse]
                        psnrs = [None, None, bpsnr, spsnr]
                        ssims = [None, None, bssim, sssim]
                        ergas = [None, None, bergas, sergas]
                        lpips = [None, None, blpips, slpips]
                        Metrics.plot_img(
                            result_imgs, mses, psnrs, ssims, ergas, lpips,
                            '{}/{}_{}_plot.png'.format(result_path, current_step, idx))

                        bic_mse += compare_mse(
                            fake_img, hr_img)
                        bic_psnr += compare_psnr(
                            fake_img, hr_img)
                        bic_ssim += compare_ssim(
                            fake_img, hr_img, multichannel=True)
                        bic_ergas += Metrics.calculate_ergas(
                            fake_img, hr_img)
                        bic_lpips += Metrics.calculate_lpips(
                            fake_img, hr_img)

                        avg_mse += compare_mse(
                            sr_img, hr_img)
                        avg_psnr += compare_psnr(
                            sr_img, hr_img)
                        avg_ssim += compare_ssim(
                            sr_img, hr_img, multichannel=True)
                        avg_ergas += Metrics.calculate_ergas(
                            sr_img, hr_img)
                        avg_lpips += Metrics.calculate_lpips(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    bic_mse = bic_mse / idx
                    bic_psnr = bic_psnr / idx
                    bic_ssim = bic_ssim / idx
                    bic_ergas = bic_ergas / idx
                    bic_lpips = bic_lpips / idx

                    avg_mse = avg_mse / idx
                    avg_psnr = avg_psnr / idx
                    avg_ssim = avg_ssim / idx
                    avg_ergas = avg_ergas / idx
                    avg_lpips = avg_lpips / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    # logger.info('# Validation # BIC_MSE: {:.2e}'.format(bic_mse))
                    # logger.info('# Validation # BIC_PSNR: {:.3e}'.format(bic_psnr))
                    # logger.info('# Validation # BIC_SSIM: {:.5e}'.format(bic_ssim))
                    # logger.info('# Validation # BIC_ERGAS: {:.4e}'.format(bic_ergas))
                    # logger.info('# Validation # BIC_LPIPS: {:.5e}'.format(bic_lpips))
                    # 
                    # logger.info('# Validation # SR_MSE: {:.2e}'.format(avg_mse))
                    # logger.info('# Validation # SR_PSNR: {:.3e}'.format(avg_psnr))
                    # logger.info('# Validation # SR_SSIM: {:.5e}'.format(avg_ssim))
                    # logger.info('# Validation # SR_ERGAS: {:.4e}'.format(avg_ergas))
                    # logger.info('# Validation # SR_LPIPS: {:.5e}'.format(avg_lpips))
                    logger_val = logging.getLogger('val')  # validation logger
                    if current_step == 8668:
                        logger_val.info(
                            'epoch: %d, iter: %d, bic_mse: %0.5e, bic_psnr: %0.5e, bic_ssim：%0.5e, bic_ergas: %0.5e, bic_lpips: %0.5e' % (
                                current_epoch, current_step, bic_mse, bic_psnr, bic_ssim, bic_ergas, bic_lpips))
                    logger_val.info(
                        'epoch: %d, iter: %d, sr_mse: %0.5e, sr_psnr: %0.5e, sr_ssim：%0.5e, sr_ergas: %0.5e, sr_lpips: %0.5e' % (
                            current_epoch, current_step, avg_mse, avg_psnr, avg_ssim, avg_ergas, avg_lpips))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        bic_mse = 0.0
        bic_psnr = 0.0
        bic_ssim = 0.0
        bic_ergas = 0.0
        bic_lpips = 0.0

        avg_mse = 0.0
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_ergas = 0.0
        avg_lpips = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            # if opt['model']['which_model_G'] == 'srcdm':
            #     diffusion.srcdm_test(continous=True)
            # else:
            #     diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8 # bicubicImg

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.tif'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                # Metrics.save_img(
                #     sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.tif'.format(result_path, current_step, idx))

            # Metrics.save_img(
            #     hr_img, '{}/{}_{}_hr.tif'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     lr_img, '{}/{}_{}_lr.tif'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     fake_img, '{}/{}_{}_inf.tif'.format(result_path, current_step, idx))

            bmse = compare_mse(fake_img, hr_img)
            bpsnr = compare_psnr(fake_img, hr_img)
            bssim = compare_ssim(fake_img, hr_img, multichannel=True)
            bergas = Metrics.calculate_ergas(fake_img, hr_img, scale=scale)
            blpips = Metrics.calculate_lpips(fake_img, hr_img)
            smse = compare_mse(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            spsnr = compare_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            sssim = compare_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img, multichannel=True)
            sergas = Metrics.calculate_ergas(Metrics.tensor2img(visuals['SR'][-1]), hr_img, scale=scale)
            slpips = Metrics.calculate_lpips(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            #result_imgs = [hr_img.cpu(), lr_img.cpu(), fake_img.cpu(), sr_img[-1].cpu()]
            result_imgs = [hr_img, lr_img, fake_img, Metrics.tensor2img(visuals['SR'][-1])]
            mses = [None, None, bmse, smse]
            psnrs = [None, None, bpsnr, spsnr]
            ssims = [None, None, bssim, sssim]
            ergas = [None, None, bergas, sergas]
            lpips = [None, None, blpips, slpips]
            Metrics.plot_img(
                result_imgs, mses, psnrs, ssims, ergas, lpips, '{}/{}_{}_plot.png'.format(result_path, current_step, idx))

            # generation
            bicmse = compare_mse(fake_img, hr_img)
            bicpsnr = compare_psnr(fake_img, hr_img)
            bicssim = compare_ssim(fake_img, hr_img, multichannel=True)
            bicergas = Metrics.calculate_ergas(fake_img, hr_img, scale=scale)
            biclpips = Metrics.calculate_lpips(fake_img, hr_img)
            eval_mse = compare_mse(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_psnr = compare_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = compare_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img, multichannel=True)
            eval_ergas = Metrics.calculate_ergas(Metrics.tensor2img(visuals['SR'][-1]), hr_img, scale=scale)
            eval_lpips = Metrics.calculate_lpips(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            bic_mse += bicmse
            bic_psnr += bicpsnr
            bic_ssim += bicssim
            bic_ergas += bicergas
            bic_lpips += biclpips
            avg_mse += eval_mse
            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_ergas += eval_ergas
            avg_lpips += eval_lpips

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)
        bic_mse = bic_mse / idx
        bic_psnr = bic_psnr / idx
        bic_ssim = bic_ssim / idx
        bic_ergas = bic_ergas / idx
        bic_lpips = bic_lpips / idx

        avg_mse = avg_mse / idx
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_ergas = avg_ergas / idx
        avg_lpips = avg_lpips / idx

        # log
        # logger.info('# Validation # BIC_MSE: {:.2e}'.format(bic_mse))
        # logger.info('# Validation # BIC_PSNR: {:.3e}'.format(bic_psnr))
        # logger.info('# Validation # BIC_SSIM: {:.5e}'.format(bic_ssim))
        # logger.info('# Validation # BIC_ERGAS: {:.4e}'.format(bic_ergas))
        # logger.info('# Validation # BIC_LPIPS: {:.5e}'.format(bic_lpips))
        #
        # logger.info('# Validation # SR_MSE: {:.2e}'.format(avg_mse))
        # logger.info('# Validation # SR_PSNR: {:.3e}'.format(avg_psnr))
        # logger.info('# Validation # SR_SSIM: {:.5e}'.format(avg_ssim))
        # logger.info('# Validation # SR_ERGAS: {:.4e}'.format(avg_ergas))
        # logger.info('# Validation # SR_LPIPS: {:.5e}'.format(avg_lpips))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info(
            'epoch: %d, iter: %d, bic_mse: %0.5e, bic_psnr: %0.5e, bic_ssim：%0.5e, bic_ergas: %0.5e, bic_lpips: %0.5e' % (
                current_epoch, current_step, bic_mse, bic_psnr, bic_ssim, bic_ergas, bic_lpips))
        logger_val.info(
            'epoch: %d, iter: %d, sr_mse: %0.5e, sr_psnr: %0.5e, sr_ssim：%0.5e, sr_ergas: %0.5e, sr_lpips: %0.5e' % (
                current_epoch, current_step, avg_mse, avg_psnr, avg_ssim, avg_ergas, avg_lpips))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
