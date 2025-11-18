import torch
from models import get_model
from pl_model.base import BasePLModel

from datasets.dataset import (
    PreprocessedSliceDataset,
    load_case_mapping_from_npy,
    split_train_val
)

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss


class SegmentationPLModel(BasePLModel):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        # 모델 생성 (2채널 output)
        self.net = get_model(self.hparams.model, channels=2)

        # Case Mapping 로드
        case_mapping = load_case_mapping_from_npy(
            preprocessed_dir=self.hparams.data_path,
            case_mapping_file=self.hparams.case_mapping_file
        )

        # Train / Val split
        train_indices, val_indices = split_train_val(
            case_mapping, train_ratio=0.8, seed=self.hparams.seed
        )
        self.train_indices = train_indices
        self.val_indices = val_indices

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        output, _, _ = self.net(x)
        return output

    # -------------------------
    # Training
    # -------------------------
    def training_step(self, batch, batch_idx):
        ct, mask = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)

        dice = self.compute_dice(output, mask)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dice, prog_bar=True)

        return loss

    # -------------------------
    # Validation
    # -------------------------
    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        output = self.forward(ct)
        loss = calc_loss(output, mask)
        dice = self.compute_dice(output, mask)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)

        return loss

    # -------------------------
    # Test
    # -------------------------
    def test_step(self, batch, batch_idx):
        ct, mask = batch
        output = self.forward(ct)
        dice = self.compute_dice(output, mask)

        self.log("test_dice", dice, prog_bar=True)
        return dice

    # -------------------------
    # DataLoader
    # -------------------------
    def train_dataloader(self):
        dataset = PreprocessedSliceDataset(
            indices=self.train_indices,
            preprocessed_dir=self.hparams.data_path,
            train=True
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self):
        dataset = PreprocessedSliceDataset(
            indices=self.val_indices,
            preprocessed_dir=self.hparams.data_path,
            train=False
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999)
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.epochs, eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        return [opt], [scheduler]
