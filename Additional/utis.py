import os
import torch
import torch.nn as nn
import lightning
import time
import pandas as pd
import torchmetrics
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
import librosa

from typing import Optional, Callable, Union


class AudioForward(nn.Module):
    def __init__(
        self,
        loss_function,
        output_key="logits",
        input_key="targets",
    ):
        super().__init__()
        self.loss_function = loss_function
        self.output_key = output_key
        self.input_key = input_key

    def forward(self, runner, batch, epoch=None):

        aus, targets = batch

        output = runner.model(aus)
        output["predictions"] = torch.argmax(output["logits"], dim=-1)

        inputs = {
            "aus": aus,
            "targets": targets,
        }

        losses = {
            "loss": self.loss_function(
                output[self.output_key],
                inputs[self.input_key],
            )
        }

        return losses, inputs, output


class LitTrainer(lightning.LightningModule):
    def __init__(
        self,
        model,
        forward,
        optimizer,
        scheduler,
        scheduler_params,
        batch_key,
        metric_input_key,
        metric_output_key,
        val_metrics,
        train_metrics=None,
    ):
        super().__init__()

        self.model = model
        self._forward = forward
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._batch_key = batch_key

        self._metric_input_key = metric_input_key
        self._metric_output_key = metric_output_key
        self._val_metrics = val_metrics
        self._train_metrics = train_metrics

    def _aggregate_outputs(self, losses, inputs, outputs):
        united = losses
        united.update({"input_" + k: v for k, v in inputs.items()})
        united.update({"output_" + k: v for k, v in outputs.items()})
        return united

    def training_step(self, batch):

        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        if self._train_metrics is not None:
            self._train_metrics.update(
                outputs[self._metric_output_key],
                inputs[self._metric_input_key]
            )

        for k, v in losses.items():
            self.log(
                "train_" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
            self.log(
                "train_avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
        self.log(
            "train_model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train_avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        return self._aggregate_outputs(losses, inputs, outputs)

    def validation_step(self, batch, batch_idx):

        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        if self._val_metrics is not None:
            self._val_metrics.update(
                outputs[self._metric_output_key],
                inputs[self._metric_input_key]
            )

        for k, v in losses.items():
            self.log(
                "valid_" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
            self.log(
                "valid_avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
        self.log(
            "valid_model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "valid_avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        return self._aggregate_outputs(losses, inputs, outputs)

    def on_train_epoch_end(self):
        metric_values = self._train_metrics.compute()
        self.log_dict(
            {"train_"+k:v for k,v in metric_values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._train_metrics.reset()

    def on_validation_epoch_end(self):
        metric_values = self._val_metrics.compute()
        self.log_dict(
            {"valid_"+k:v for k,v in metric_values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._val_metrics.reset()

    def configure_optimizers(self):
        scheduler = {"scheduler": self._scheduler}
        scheduler.update(self._scheduler_params)
        return (
            [self._optimizer],
            [scheduler],
        )


def lightning_training(
    train_df: Optional[pd.DataFrame],
    val_df: Optional[pd.DataFrame],
    exp_name: str,
    fold_id: Optional[int],
    forward_batch_key: str,
    train_dataset_class: torch.utils.data.Dataset,
    val_dataset_class: Optional[torch.utils.data.Dataset],
    train_dataset_config: dict,
    val_dataset_config: Optional[dict],
    train_dataloader_config: dict,
    val_dataloader_config: Optional[dict],
    nn_model_class: torch.nn.Module,
    nn_model_config: dict,
    optimizer_init: Callable,
    scheduler_init: Callable,
    scheduler_params: dict,
    forward: Union[torch.nn.Module, Callable],
    n_epochs: int,
    main_metric: str,
    metric_mode: str,
    metric_input_key: str,
    metric_output_key: str,
    val_metrics: torchmetrics.MetricCollection,
    train_metrics: Optional[torchmetrics.MetricCollection] = None,
    # It is not really Callable. It just lambda that will init List of callbacks
    # each time. It is just done for safe CV training.
    callbacks: Optional[Callable] = None,
    checkpoint_callback_params: dict = {},
    wandb_logger_params: dict = {},
    trainer_params: dict = {},
    precision_mode: str = "32-true",
    n_checkpoints_to_save: int = 3,
    log_every_n_steps: int = 3,
    train_strategy: str = "auto",
):
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training Device : {device}")

    train_dataset = train_dataset_class(
        input_df=train_df,
        **train_dataset_config,
    )
    val_dataset = val_dataset_class(
        input_df=val_df,
        **val_dataset_config,
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            **train_dataloader_config,
        ),
        "valid": torch.utils.data.DataLoader(
            val_dataset, 
            **val_dataloader_config
        )
    }

    model = nn_model_class(device=device, **nn_model_config)

    for k in loaders.keys():
        print(f"{k} Loader Len = {len(loaders[k])}")

    optimizer = optimizer_init(model)
    scheduler = scheduler_init(optimizer, len(loaders["train"]))

    if not isinstance(forward, torch.nn.Module):
        forward = forward()

    lightning_model = LitTrainer(
        model,
        forward=forward,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        batch_key=forward_batch_key,
        metric_input_key=metric_input_key,
        metric_output_key=metric_output_key,
        val_metrics=val_metrics,
        train_metrics=train_metrics
    )

    all_callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(exp_name, "checkpoints"),
            save_top_k=n_checkpoints_to_save,
            mode=metric_mode,
            monitor=main_metric,
            **checkpoint_callback_params,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if callbacks is not None:
        all_callbacks += callbacks()

    wandb_logger = pl_loggers.WandbLogger(
        save_dir=exp_name,
        name=exp_name,
        **wandb_logger_params,
    )
    trainer = lightning.Trainer(
        devices=-1,
        precision=precision_mode,
        strategy=train_strategy,
        max_epochs=n_epochs,
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=all_callbacks,
        **trainer_params,
    )
    trainer.fit(model=lightning_model, train_dataloaders=loaders["train"], val_dataloaders=loaders["valid"])
    return all_callbacks[0].best_model_path


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_df,
        filenpath_col="filepath",
        target_col="target",
        sample_rate=16000,
        normalize_audio=True,
        audio_transforms=None,
    ):        
        self.df = input_df.reset_index(drop=True)

        self.filenpath_col = filenpath_col
        self.target_col = target_col

        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

        self.audio_transforms = audio_transforms

    def __len__(self):
        return len(self.df)

    def _prepare_sample(self, idx: int):
        au, sr = librosa.load(self.df[self.filenpath_col].iloc[idx], sr=self.sample_rate)
        assert sr == self.sample_rate
        # We know that all samples are of the same length = 5 sec and contains only one channel
        assert len(au.shape) == 1
        assert au.shape[0] == self.sample_rate * 5

        target_idx = self.df[self.target_col].iloc[idx]

        if self.audio_transforms is not None:
            au = self.audio_transforms(samples=au, sample_rate=sr)

        if self.normalize_audio:
            au = librosa.util.normalize(au)

        return torch.from_numpy(au).float(), target_idx

    def __getitem__(self, idx: int):
        return self._prepare_sample(idx)