import os
import wandb

from configs import get_train_config, setup_seed
from src.datasets import get_loader
from src.models import create_model
from src.trainers import get_trainer

if __name__ == '__main__':
    # seed
    setup_seed(0)

    # config
    train_cfg = get_train_config("configs/train.yaml")

    # wandb
    model_name = train_cfg.model_name
    marine_param = train_cfg.marine_param
    upscale = train_cfg.upscale
    os.environ["WANDB_MODE"] = "offline"
    wandb_log = wandb.init(
        project="Marine-Parameters-SupervisedSR",
        name=f"{marine_param}-x{upscale}-{model_name}"
    )

    # loader
    train_loader, val_loader = get_loader(train_cfg, batch_size=train_cfg.batch_size)

    # model
    model = create_model(train_cfg)

    # trainer
    trainer = get_trainer(
        train_cfg=train_cfg,
        model_dict=model,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_log=wandb_log
    )
    trainer.load_weight(train_cfg.checkpoint)
    trainer.fit()
    wandb.finish()