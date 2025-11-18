import os
import argparse

from pl_model.knowledge_distillation_model import KnowledgeDistillationPLModel

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


parser = argparse.ArgumentParser('train_kd')
parser.add_argument('--data_path', type=str, default='/data/kits/data')
parser.add_argument('--checkpoint_path', type=str, default='/data/checkpoints')
parser.add_argument('--tckpt', type=str, default='/data/checkpoints/checkpoint_kits_tumor_enet_epoch=18.ckpt', help='teacher model checkpoint path')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--smodel', type=str, default='enet')
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'organ'])
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='kits', choices=['kits', 'lits'])

def main():
    args = parser.parse_args()
    seed_everything(args.seed)
    
    model = KnowledgeDistillationPLModel(args)

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_path),
        filename='checkpoint_%s_%s_kd_%s_{epoch}' % (args.dataset, args.task, args.smodel),
        save_last=True,
        save_top_k=5,
        monitor='dice_class0',
        mode='max',
        verbose=True
    )

    logger = TensorBoardLogger('log', name='%s_%s_kd_%s' % (args.dataset, args.task, args.smodel))
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs, 
        callbacks=[checkpoint_callback], 
        logger=logger
    )
    trainer.fit(model)


def test():
    args = parser.parse_args()
    model = KnowledgeDistillationPLModel.load_from_checkpoint(
        checkpoint_path=os.path.join(args.checkpoint_path, 'last.ckpt')
    )

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
    )
    trainer.test(model)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        main()
    if args.mode == 'test':
        test()