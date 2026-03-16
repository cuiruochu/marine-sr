import os
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.testers import UnifiedModelCommonTester


class UnifiedModelCommonTrainer(UnifiedModelCommonTester):
    def __init__(self, train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log):
        super(UnifiedModelCommonTrainer, self).__init__(val_cfg, model_dict, val_loader)
        # train
        self.set_train()
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=epoches // 2, gamma=0.1)
        self.epoches = epoches
        self.start_epoch = 1
        self.pth_save_path = train_cfg.pth_save_path
        # loader
        self.train_loader = train_loader
        # log
        self.wandb_log = wandb_log

    def print_model_info(self):
        print("Model Name:", self.model_name)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.mwd_encoder.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.other_encoder.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.mwd_decoder.parameters() if p.requires_grad)
        total_params += sum(p.numel() for p in self.other_decoder.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

    def set_train(self):
        self.mwd_encoder.train()
        self.other_encoder.train()
        self.mwd_decoder.train()
        self.other_decoder.train()
        self.model.train()

    def load_weight(self, checkpoint):
        if not os.path.exists(checkpoint):
            print("The file path for the model parameters does not exist and will be trained from scratch...")
            return
        print(f'===> Loading pre-trained parameters: {checkpoint}')
        checkpoint_dict = torch.load(checkpoint, weights_only=True)
        self.model.load_state_dict(checkpoint_dict["model"], strict=True)
        self.mwd_encoder.load_state_dict(checkpoint_dict["mwd_encoder"], strict=True)
        self.other_encoder.load_state_dict(checkpoint_dict["other_encoder"], strict=True)
        self.mwd_decoder.load_state_dict(checkpoint_dict["mwd_decoder"], strict=True)
        self.other_decoder.load_state_dict(checkpoint_dict["other_decoder"], strict=True)
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
        self.start_epoch = checkpoint_dict['epoch'] + 1

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.pth_save_path, self.model_name + f"_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'mwd_encoder': self.mwd_encoder.state_dict(),
            'other_encoder': self.other_encoder.state_dict(),
            'mwd_decoder': self.mwd_decoder.state_dict(),
            'other_decoder': self.other_decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, save_path)

    def train_epoch(self, epoch):
        self.model.train()
        for iteration, batch in enumerate(tqdm(self.train_loader), 1):
            LR_img, HR_img = batch[0].to(self.device), batch[1].to(self.device)
            marine_param_idx = batch[2][0]
            # model inference
            LR_feats = self.mwd_encoder(LR_img) if marine_param_idx == 1 else self.other_encoder(LR_img)
            SR_feats = self.model(LR_feats)
            SR_img = self.mwd_decoder(SR_feats) if marine_param_idx == 1 else self.other_decoder(SR_feats)
            # loss
            loss = self.criterion(SR_img, HR_img)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # wandb log
            self.wandb_log.log({
                "loss": loss.data,
                "learning_rate": self.scheduler.get_last_lr()[0],
            })
            # print
            if iteration % 500 == 0:
                print(f"===> Epoch[{epoch}]({iteration}/{len(self.train_loader)}): Loss:{loss.data:.4f}")
        self.scheduler.step()

    def eval_epoch(self, epoch):
        avg_psnr, avg_ssim, avg_mae = self.eval()
        return avg_psnr, avg_ssim, avg_mae

    def fit(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epoches):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
