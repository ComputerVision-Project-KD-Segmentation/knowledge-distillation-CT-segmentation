import torch
from pytorch_lightning.core import LightningModule

class BasePLModel(LightningModule):
    def __init__(self):
        super(BasePLModel, self).__init__()
        self.metric = []

    def training_epoch_end(self, outputs):
        train_loss_mean = sum([o['loss'] for o in outputs]) / len(outputs)
        self.log('train_loss', train_loss_mean)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def measure(self, batch, output):
        ct, mask = batch   # (B,1,H,W), (B,2,H,W)

        # -------------------------
        # prediction foreground
        # -------------------------
        prob = torch.softmax(output, dim=1)[:, 1]    # (B,H,W)

        # -------------------------
        # ground truth foreground
        # -------------------------
        gt = mask[:, 1]  # (B,H,W) ← 반드시 channel=1 !!!

        # -------------------------
        # threshold
        # -------------------------
        pred = (prob > 0.5).float()

        # -------------------------
        # accumulate per-slice dice components
        # -------------------------
        for p, g in zip(pred, gt):
            inter = (p * g).sum()
            pre = p.sum()
            gt_ = g.sum()

            # store raw numbers
            self.metric.append((
                inter.item(),
                pre.item(),
                gt_.item()
            ))

    def test_epoch_end(self, outputs):
        inter_sum = 0
        count = 0

        for inter, pre, gt in self.metric:
            if gt > 0:
                dice = (2 * inter + 1) / (pre + gt + 1)
                inter_sum += dice
                count += 1

        final_dice = inter_sum / max(count, 1)

        self.log('dice', final_dice)
        print(f"\n[DICE]: {final_dice:.4f}")

        self.metric = []
