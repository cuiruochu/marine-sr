from .common_trainer import CommonTrainer
from .unified_trainer import UnifiedModelCommonTrainer
from .CAMixer import TrainerCAMixer


def get_trainer(train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log):
    if train_cfg.unified:
        Trainer = UnifiedModelCommonTrainer(
            train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log)
    else:
        if train_cfg.model_name == "CAMixer":
            Trainer = TrainerCAMixer(train_cfg, model_dict, lr, epoches, train_loader, val_loader, wandb_log)
        else:
            Trainer = CommonTrainer(train_cfg, model_dict, lr, epoches, train_loader, val_cfg, val_loader, wandb_log)
    return Trainer
