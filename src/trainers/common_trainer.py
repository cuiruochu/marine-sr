import os
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR, StepLR
from src.testers import CommonTester


class CommonTrainer(CommonTester):
    def __init__(self, train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log):
        super(CommonTrainer, self).__init__(val_cfg, model_dict, val_loader)
        # model
        self.model.train()
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=epoches // 2, gamma=0.1)
        self.epoches = epoches
        self.start_epoch = 1
        self.pth_save_path = train_cfg.pth_save_path
        # loader配置
        self.train_loader = train_loader
        # log配置
        self.wandb_log = wandb_log

    def find_lr(self, init_value=1e-8, final_value=10., beta=0.98):
        print("开始搜索学习率...")

        num = len(self.train_loader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        lrs = []
        for data in tqdm(self.train_loader):
            batch_num += 1
            # As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # inputs, labels = Variable(inputs), Variable(labels)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            lrs.append(lr)
            # Do the SGD step
            loss.backward()
            self.optimizer.step()
            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr
        return lrs, losses

    def load_weight(self, checkpoint):
        if not os.path.exists(checkpoint):
            print("The file path for the model parameters does not exist and will be trained from scratch...")
            return
        print(f'===> Loading pre-trained parameters: {checkpoint}')
        checkpoint_dict = torch.load(checkpoint, weights_only=True)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self.scheduler.load_state_dict(checkpoint_dict['scheduler'])
        self.start_epoch = checkpoint_dict['epoch'] + 1

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.pth_save_path, self.model_name + f"_epoch{epoch}.pth")
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, save_path)

    def train_epoch(self, epoch):
        self.model.train()
        for iteration, batch in enumerate(tqdm(self.train_loader), 1):
            LR_img, HR_img = batch[0].to(self.device), batch[1].to(self.device)
            # model forward
            SR_img = self.model(LR_img)
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
            if iteration % 200 == 0:
                print(f"===> Epoch[{epoch}]({iteration}/{len(self.train_loader)}): Loss:{loss.item():.4f}")
        self.scheduler.step()

    def eval_epoch(self, epoch):
        eval_psnr, eval_ssim, eval_mae = self.eval()
        # wandb log
        self.wandb_log.log({
            "epoch": epoch,
            "eval-mae": eval_mae,
            "eval-psnr": eval_psnr,
            "eval-ssim": eval_ssim,
        })
        return eval_mae, eval_psnr, eval_ssim

    def fit(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.epoches):
            self.train_epoch(epoch)
            self.eval_epoch(epoch)
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
