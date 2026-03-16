from .common_trainer import CommonTrainer
from .CAMixer import TrainerCAMixer


def get_trainer(train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log):
    if train_cfg.model_name == "CAMixer":
        return TrainerCAMixer(train_cfg, model_dict, lr, epoches, train_loader, val_loader, wandb_log)
    else:
        return CommonTrainer(train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log)