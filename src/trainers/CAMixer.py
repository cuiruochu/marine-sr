from .common_trainer import *


class TrainerCAMixer(CommonTrainer):
    def __init__(self, config, model_dict, lr, epoches, train_loader, val_loader, wandb_log):
        super(TrainerCAMixer, self).__init__(config, model_dict, lr, epoches, train_loader, val_loader, wandb_log)

    def train_one_epoch(self, epoch):
        epoch_loss = 0
        self.model.train()

        for iteration, batch in enumerate(tqdm(self.train_loader), 1):
            t0 = time.time()
            LR_img = batch[0].to(self.device, non_blocking=True)
            HR_img = batch[1].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            SR_img, loss_ratio = self.model(LR_img)
            loss_pix = self.criterion(SR_img, HR_img)
            loss = loss_pix + loss_ratio
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.data
            t1 = time.time()

            # wandb log
            self.wandb_log.log({
                "loss": loss.data,
                "loss_pix": loss_pix.data,
                "loss_ratio": loss_ratio,
                "learning_rate": self.scheduler.get_last_lr()[0],
            })

            # print training information every 100 iterations
            if iteration % 100 == 0:
                print(f"===> Epoch[{epoch}]({iteration}/{len(self.train_loader)}):"
                      f" Loss:{loss.data:.4f} || Time:{t1 - t0:.4f} sec.")

        train_loss = epoch_loss / len(self.train_loader)
        tqdm.write(f"===> Epoch {epoch}: Avg.Loss:{train_loss:.4f}")
        self.scheduler.step()

        return train_loss

