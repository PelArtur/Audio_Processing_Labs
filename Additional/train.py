import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torchmetrics
from utis import lightning_training 
from utis import AudioDataset
from model import SpecCNNClasifier
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset_class = AudioDataset
    val_dataset_class = AudioDataset

    nn_model_class = SpecCNNClasifier
    nn_model_config = cfg.model.nn_model_config

    nn_model_config["n_classes"] = cfg.train.n_classes

    def optimizer_init(model):
        return Adam(model.parameters(), lr=cfg.optimizer.lr)

    def scheduler_init(optimizer, steps_per_epoch):
        return CosineAnnealingLR(optimizer, T_max=cfg.scheduler.t_max, eta_min=cfg.scheduler.eta_min)

    # train_df = pd.read_csv("train_meta.csv")
    # val_df = pd.read_csv("val_meta.csv")

    val_metrics = torchmetrics.MetricCollection([
        torchmetrics.classification.MulticlassF1Score(num_classes=cfg.train.n_classes, average="macro")
    ])
    train_metrics = torchmetrics.MetricCollection([
        torchmetrics.classification.MulticlassF1Score(num_classes=cfg.train.n_classes, average="macro")
    ])

    print("Parameters from yaml files")
    for val in cfg['exp']:
        print(val)
    lightning_training(
        train_df=pd.DataFrame(),           #dummy
        val_df=pd.DataFrame(),             #dummy  
        exp_name=cfg['exp']["exp_name"],
        fold_id=cfg['exp']["fold_id"],
        forward_batch_key=cfg['train']["forward_batch_key"],
        train_dataset_class=train_dataset_class,
        val_dataset_class=val_dataset_class,
        train_dataset_config=cfg['dataset']["train_dataset_config"],
        val_dataset_config=cfg['dataset']["val_dataset_config"],
        train_dataloader_config=cfg['dataloader']["train_dataloader_config"],
        val_dataloader_config=cfg['dataloader']["train_dataloader_config"],
        nn_model_class=nn_model_class,
        nn_model_config=nn_model_config,
        optimizer_init=optimizer_init,
        scheduler_init=scheduler_init,
        n_epochs=cfg['train']["n_epochs"],
        main_metric=cfg['train']["main_metric"],
        metric_mode=cfg['train']["metric_mode"],
        metric_input_key=cfg['train']["metric_input_key"],
        metric_output_key=cfg['train']["metric_output_key"],
        val_metrics=val_metrics,
        train_metrics=train_metrics,
        callbacks=None,
        checkpoint_callback_params=cfg['train']["checkpoint_callback_params"],
        wandb_logger_params=cfg['exp']["wandb_logger_params"],
        precision_mode=cfg['exp']["precision_mode"],
        scheduler_params=None,  #dummy
        forward=None  #dummy
    )

if __name__ == "__main__":
    main()
