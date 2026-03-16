import torchvision.transforms as transforms

from .common_tester import *
from .utils import reverse_norm, apply_mask, normalize_to_01, \
    calculate_mae_pt, calculate_max_mae_pt, calculate_psnr_pt, calculate_ssim_pt


class TesterBicubic(CommonTester):
    def __init__(self, config, model_dict, test_loader):
        super(TesterBicubic, self).__init__(config, model_dict, test_loader)
        self.upscale = config.upscale

    def print_model_info(self):
        print("Model Name:", self.model_name)

    def load_model(self, *args):
        pass

    def eval(self):
        total_psnr, total_ssim, total_mae, max_mae = 0, 0, 0, 0
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(self.test_loader), 1):
                LR_img, HR_img = batch[0].to(self.device), batch[1].to(self.device)
                LR_img = reverse_norm(LR_img, mean=self.mean, std=self.std)
                resize_fn = transforms.Resize(HR_img.shape[-2:], transforms.InterpolationMode.BICUBIC)
                SR_img = resize_fn(LR_img)
                # mask
                mask_matrix = self.mask_matrix.expand_as(HR_img) if self.mask_matrix is not None else None
                HR_img, SR_img = apply_mask(mask_matrix, HR_img, SR_img)
                HR_norm, SR_norm = normalize_to_01(HR_img, SR_img)
                # metrics
                total_mae += calculate_mae_pt(SR_img, HR_img, mask=mask_matrix)
                max_tmp = calculate_max_mae_pt(SR_img, HR_img, mask=mask_matrix)
                if max_tmp > max_mae:
                    max_mae = max_tmp
                total_psnr += calculate_psnr_pt(SR_norm, HR_norm, mask=mask_matrix)
                total_ssim += calculate_ssim_pt(SR_norm, HR_norm, mask=mask_matrix)
        avg_mae = total_mae / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        tqdm.write(f"===> Test: PSNR:{avg_psnr:.4f}; SSIM:{avg_ssim:.4f}; MAE:{avg_mae:.4f}; MMae: {max_mae:.4f}")
        return avg_psnr, avg_ssim, avg_mae

    def save_results(self, save_path, makedir=False):
        if not os.path.exists(save_path) and not makedir:
            raise FileNotFoundError("SR的保存路径似乎不存在，如果需要创建，请指定makedir=True...")
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for iter, batch in enumerate(tqdm(self.test_loader), 1):
                # lr: (1, C, H ,W), filename_tuple:(1,)
                lr, hr, filename_tuple = batch[0].to(self.device), batch[1], batch[2]
                assert lr.shape[0] == 1, "仅支持batch_size=1"
                resize_fn = transforms.Resize(hr.shape[-2:], transforms.InterpolationMode.BICUBIC)
                sr = resize_fn(lr)
                sr = reverse_norm(sr, mean=self.mean, std=self.std)
                mask_matrix = self.mask_matrix.expand_as(sr) if self.mask_matrix is not None else None
                sr = apply_mask(mask_matrix, sr)
                # 保存SR为npy文件: (C, H, W)
                sr = sr.squeeze(0)
                filename, _ = os.path.splitext(filename_tuple[0])
                file_save_path = os.path.join(save_path, filename + '.npy')
                np.save(file_save_path, sr.cpu().numpy())

    def extrapolate(self, save_path, makedir=False):
        if not os.path.exists(save_path) and not makedir:
            raise FileNotFoundError("SR的保存路径似乎不存在，如果需要创建，请指定makedir=True...")
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for iter, batch in enumerate(tqdm(self.test_loader), 1):
                lr, hr, filename_tuple = batch[0], batch[1], batch[2]
                resize_fn = transforms.Resize(
                    (hr.shape[-2] * self.upsacle, hr.shape[-1] * self.upsacle),
                    transforms.InterpolationMode.BICUBIC
                )
                extrapolation = resize_fn(hr).squeeze(0)
                extrapolation = self.reverse_norm(extrapolation)
                # 保存SR为npy文件
                filename, _ = os.path.splitext(filename_tuple[0])
                file_save_path = os.path.join(save_path, filename + '.npy')
                np.save(file_save_path, extrapolation.cpu().numpy())
