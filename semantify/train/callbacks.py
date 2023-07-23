import os
import json
import shutil
import pytorch_lightning as pl
from typing import Optional
from pytorch_lightning.callbacks import Callback
from semantify.utils.paths_utils import append_to_root_dir


class CreateModelMeta(Callback):
    def __init__(self, outpath: Optional[str] = ""):
        self.out_path = append_to_root_dir("pre_production") if outpath == "" else outpath

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path is None:
            return
        ckpt_new_name = f"{trainer.logger.name}.ckpt"
        ckpt_new_path = ckpt_path.replace(ckpt_path.split("/")[-1], ckpt_new_name)
        os.rename(ckpt_path, ckpt_new_path)
        shutil.copy(ckpt_new_path, f"{self.out_path}/{ckpt_new_name}")
        pl_module.hparams.hidden_size = list(pl_module.hparams.hidden_size)
        metadata = dict(pl_module.hparams)
        with open(f"{self.out_path}/{ckpt_new_path.split('/')[-1].replace('.ckpt', '_metadata.json')}", "w") as f:
            json.dump(metadata, f)
