import torch
from models import get_model
from pl_model.base import BasePLModel
from pl_model.segmentation_model import SegmentationPLModel
from utils.loss_functions import prediction_map_distillation, importance_maps_distillation, region_affinity_distillation
from datasets.dataset import SliceDataset, load_case_mapping, split_train_val, get_data_list

from torch.utils.data import DataLoader
from utils.loss_functions import calc_loss

# KD loss para
alpha = 0.1
beta1 = 0.9
beta2 = 0.9

class KnowledgeDistillationPLModel(BasePLModel):
    def __init__(self, params):
        super(KnowledgeDistillationPLModel, self).__init__()
        self.save_hyperparameters(params)

        # load and freeze teacher net
        self.t_net = SegmentationPLModel.load_from_checkpoint(checkpoint_path=self.hparams.tckpt)
        self.t_net.freeze()

        # student net
        self.net = get_model(self.hparams.smodel, channels=2)

        case_mapping = load_case_mapping(self.hparams.data_path)
        train_indices, val_indices = split_train_val(
            case_mapping, train_ratio=0.8, seed=self.hparams.seed
        )
        self.train_indices = train_indices
        self.val_indices = val_indices

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ct, mask, name = batch
        self.t_net.eval()
        t_out, t_low, t_high = self.t_net.net(ct)
        output, low, high, = self.net(ct)

        loss_seg = calc_loss(output, mask)

        loss_pmd = prediction_map_distillation(output, t_out)
        loss_imd = importance_maps_distillation(low, t_low) + importance_maps_distillation(high, t_high)
        loss_rad = region_affinity_distillation(low, t_low, mask) + region_affinity_distillation(high, t_high, mask)

        loss = loss_seg + alpha * loss_pmd + beta1 * loss_imd + beta2 * loss_rad

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        ct, mask, name = batch
        output, low, high = self.net(ct)

        self.measure(batch, output)

    def train_dataloader(self):
        data_list = get_data_list(self.hparams.data_path, self.train_indices)
        dataset = SliceDataset(
            data_list=data_list,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=True
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        data_list = get_data_list(self.hparams.data_path, self.val_indices)
        dataset = SliceDataset(
            data_list=data_list,
            task=self.hparams.task,
            dataset=self.hparams.dataset,
            train=False
        )
        return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return self.test_dataloader()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999))
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-6),
                     'interval': 'epoch',
                     'frequency': 1}
        return [opt], [scheduler]