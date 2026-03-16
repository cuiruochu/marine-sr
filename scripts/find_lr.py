from configs import get_train_config
from src.datasets import get_train_loader
from src.trainers import get_trainer


def plot_lr_loss(log_lrs, losses):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(log_lrs, losses)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    plt.xscale('log')
    plt.show()


if __name__ == '__main__':
    train_cfg = get_train_config("configs/train.yaml")
    train_loader, val_loader = get_train_loader(train_cfg, batch_size=train_cfg.batch_size)

    trainer_cls = get_trainer(train_cfg)
    trainer = trainer_cls(
        train_cfg=train_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    lrs, losses = trainer.find_lr()
    plot_lr_loss(lrs, losses)