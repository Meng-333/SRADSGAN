import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        # default='experiments/basic_sr_ffhq_210809_142238/results')
                        default='experiments/sr_mfe_221223_180956_105937/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    bic_names = list(glob.glob('{}/*_inf.png'.format(args.path)))

    real_names.sort()
    fake_names.sort()
    bic_names.sort()

    scale = 4

    # str1 = '64'
    # str2 = '32'
    # if str1 in args.path:
    #     scale = 4
    # if str2 in args.path:
    #     scale = 8
    # print('args.path:', args, args.path, 'scale:', scale)

    # bic
    bic_mse = 0.0
    bic_psnr = 0.0
    bic_ssim = 0.0
    bic_ergas = 0.0
    bic_lpips = 0.0
    # sr
    avg_mse = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_ergas = 0.0
    avg_lpips = 0.0
    idx = 0
    for rname, fname, bicname in zip(real_names, fake_names, bic_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = rname.rsplit("_sr")[0]
        bicidx = rname.rsplit("_inf")[0]
        assert ridx == fidx == bicidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
            ridx, fidx)

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        bic_img = np.array(Image.open(bicname))

        bicmse = Metrics.calculate_mse(bic_img, hr_img)
        bicpsnr = Metrics.calculate_psnr(bic_img, hr_img)
        bicssim = Metrics.calculate_ssim(bic_img, hr_img)
        bicergas = Metrics.calculate_ergas(bic_img, hr_img, scale=scale)
        biclpips = Metrics.calculate_lpips(bic_img, hr_img)
        bic_mse += bicmse
        bic_psnr += bicpsnr
        bic_ssim += bicssim
        bic_ergas += bicergas
        bic_lpips += biclpips

        mse = Metrics.calculate_mse(sr_img, hr_img)
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        ergas = Metrics.calculate_ergas(sr_img, hr_img, scale=scale)
        lpips = Metrics.calculate_lpips(sr_img, hr_img)
        avg_mse += mse
        avg_psnr += psnr
        avg_ssim += ssim
        avg_ergas += ergas
        avg_lpips += lpips
        # if idx % 20 == 0:
        #     print('The Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(idx, psnr, ssim))

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
    print('# Validation # BIC_MSE: {:.2e}'.format(bic_mse))
    print('# Validation # BIC_PSNR: {:.3e}'.format(bic_psnr))
    print('# Validation # BIC_SSIM: {:.5e}'.format(bic_ssim))
    print('# Validation # BIC_ERGAS: {:.4e}'.format(bic_ergas))
    print('# Validation # BIC_LPIPS: {:.5e}'.format(bic_lpips))

    print('# Validation # SR_MSE: {:.2e}'.format(avg_mse))
    print('# Validation # SR_PSNR: {:.3e}'.format(avg_psnr))
    print('# Validation # SR_SSIM: {:.5e}'.format(avg_ssim))
    print('# Validation # SR_ERGAS: {:.4e}'.format(avg_ergas))
    print('# Validation # SR_LPIPS: {:.5e}'.format(avg_lpips))
