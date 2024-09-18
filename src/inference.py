# inference.py
from typing import List
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
from src.datamodules.inference_dataloader import PNGDataset
from src import utils

log = utils.get_pylogger(__name__)

def inference(cfg: DictConfig) -> List:
    """Performs inference using the given configuration.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        List: Predictions from the model.
    """
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting inference!")
    dataset = PNGDataset(cfg.data_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    predictions = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=cfg.ckpt_path)

    return predictions

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference_objects.yaml")
def main(cfg: DictConfig) -> None:
    predictions = inference(cfg)
    # Handle predictions as needed, e.g., save to file or process further
    print(predictions)

if __name__ == "__main__":
    main()