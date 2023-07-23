import time
import torch
import hydra
import logging
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from semantify.train.mapper_module import C2M_pl
from semantify.train.callbacks import CreateModelMeta
from semantify.train.dataset import semantifyDataset
from semantify.utils.paths_utils import append_to_root_dir

OmegaConf.register_new_resolver("append_to_root_dir", append_to_root_dir)


def train(config: DictConfig):

    start_time = time.time()

    seed_everything(config.seed)

    assert config.tensorboard_logger.name is not None, "must specify a suffix"

    log = logging.getLogger(__name__)

    dataset = semantifyDataset(**config.dataset)
    train_size = int(len(dataset) * config.train_size)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    callbacks = [CreateModelMeta(config.output_path)]

    log.info(f"dataloader batch size: {config.dataloader.batch_size}")
    train_dataloader = DataLoader(train_dataset, **config.dataloader)
    val_dataloader = DataLoader(val_dataset, **config.dataloader)

    log.info(f"tensorboard run name: {config.tensorboard_logger.name}")
    logger = TensorBoardLogger(**config.tensorboard_logger)

    log.info(f"instantiating model")
    trainer = Trainer(logger=logger, callbacks=callbacks, **config.trainer)

    num_stats = config.dataset.labels_to_get.__len__()

    model = C2M_pl(num_stats=num_stats, **config.model_conf, labels=dataset.get_labels())

    log.info(f"training model")
    trainer.fit(model, train_dataloader, val_dataloader)

    log.info(f"finished training in {time.time() - start_time} seconds")


@hydra.main(config_path="../config", config_name="train_mapper")
def main(config: DictConfig) -> None:
    train(config)


if __name__ == "__main__":
    main()
