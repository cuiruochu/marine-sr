import os
import torch
import numpy as np
from tqdm import tqdm

from .utils import reverse_norm, apply_mask, normalize_to_01, \
    calculate_mae_pt, calculate_max_mae_pt, calculate_psnr_pt, calculate_ssim_pt


class CommonTester:
    def __init__(self, config, model_dict, test_loader):
        # Model配置
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if model_dict['model']:
            self.model = model_dict['model'].to(self.device).eval()
        self.model_name = model_dict['model_name']
        self.print_model_info()
        # loader配置
        self.test_loader = test_loader
        # norm
        self.mean = torch.tensor(config.mean).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.std = torch.tensor(config.std).unsqueeze(-1).unsqueeze(-1).to(self.device)
        if config.eval_mask:
            self.mask_matrix = torch.from_numpy(np.load(config.eval_mask)).to(self.device).unsqueeze(0).unsqueeze(0)
            print("当前对 指标 计算时对HR进行mask!!!")
        else:
            self.mask_matrix = None
            print("当前对 指标 计算时不mask!!!")

    def print_model_info(self):
        print("Model Name:", self.model_name)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def load_model(self, checkpoint):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError("The file path for the model parameters does not exist...")
        print(f'===> Loading pre-trained parameters: {checkpoint}')
        checkpoint_dict = torch.load(checkpoint, weights_only=True)
        self.model.load_state_dict(checkpoint_dict["model"], strict=True)

    def eval(self):
        self.model.eval()
        total_psnr, total_ssim, total_mae, max_mae = 0, 0, 0, 0
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(self.test_loader), 1):
                LR_img, HR_img = batch[0].to(self.device), batch[1].to(self.device)
                SR_img = self.model(LR_img)
                # norm
                SR_img = reverse_norm(SR_img, mean=self.mean, std=self.std)
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
        tqdm.write(f"===> Test: PSNR:{avg_psnr:.4f}; SSIM:{avg_ssim:.4f}; MAE:{avg_mae:.4f}; MMae:{max_mae:.4f}")
        return avg_psnr, avg_ssim, avg_mae

    def save_results(self, save_path, makedir=False):
        if not os.path.exists(save_path) and not makedir:
            raise FileNotFoundError("SR的保存路径似乎不存在，如果需要创建，请指定makedir=True...")
        os.makedirs(save_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(self.test_loader), 1):
                # lr: (1, C, H ,W), filename_tuple:(1,)
                lr, filename_tuple = batch[0].to(self.device), batch[2]
                assert lr.shape[0] == 1, "仅支持batch_size=1"
                sr = self.model(lr)
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
                # hr: (1, 1, H ,W), filename_tuple:(1,)
                hr, filename_tuple = batch[1].to(self.device), batch[2]
                # extrapolation: (C, H, W)
                extrapolation = self.model(hr).squeeze(0)
                extrapolation = self.reverse_norm(extrapolation)
                # 保存SR为npy文件
                filename, _ = os.path.splitext(filename_tuple[0])
                file_save_path = os.path.join(save_path, filename + '.csv')
                np.save(file_save_path, extrapolation.cpu().numpy())
