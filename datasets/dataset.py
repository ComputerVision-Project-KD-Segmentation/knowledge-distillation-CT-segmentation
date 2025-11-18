# ============================================
# dataset.py — Preprocessed KiTS19 (.npy) 버전
# ============================================

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict


# ============================================
# 1) Dataset 클래스
# ============================================

class PreprocessedSliceDataset(Dataset):
    def __init__(self, indices, preprocessed_dir, train=True):
        self.indices = indices
        self.train = train

        # Load big arrays
        self.images = np.load(os.path.join(preprocessed_dir, "images.npy"))
        self.masks  = np.load(os.path.join(preprocessed_dir, "masks.npy"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        img = self.images[index]     # (1,H,W)
        msk = self.masks[index]      # (2,H,W)

        if self.train:
            img, msk = self.augment(img, msk)

        img = torch.from_numpy(img).float()
        msk = torch.from_numpy(msk).float()

        return img, msk

    def augment(self, img, msk):
        # Horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=2).copy()

        # Vertical flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=1).copy()

        # Rotate 90/180/270
        if random.random() < 0.5:
            k = random.randint(1, 3)
            img = np.rot90(img, k, axes=(1,2)).copy()
            msk = np.rot90(msk, k, axes=(1,2)).copy()

        return img, msk



# ============================================
# 2) Case mapping (npy version)
# ============================================

def load_case_mapping_from_npy(preprocessed_dir, case_mapping_file):
    """
    전처리된 슬라이스들은 case_id 정보가 없기 때문에,
    학습 시 동일한 fold를 만들기 위해 기존 case_mapping.pkl을 로드.
    """

    import pickle
    with open(case_mapping_file, "rb") as f:
        case_mapping = pickle.load(f)

    # case_mapping contains slice indices already
    return case_mapping


# ============================================
# 3) Train/Val Split (case-level)
# ============================================

def split_train_val(case_mapping, train_ratio=0.8, seed=42):
    case_ids = list(case_mapping.keys())
    random.seed(seed)
    random.shuffle(case_ids)

    n_train = int(len(case_ids) * train_ratio)
    train_cases = case_ids[:n_train]
    val_cases = case_ids[n_train:]

    train_idx = []
    val_idx = []

    for c in train_cases:
        train_idx.extend(case_mapping[c]["indices"])

    for c in val_cases:
        val_idx.extend(case_mapping[c]["indices"])

    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)

    print(f"\n[Split] Case-level split 완료")
    print(f"  Train: {len(train_cases)} cases, {len(train_idx):,} slices")
    print(f"  Val:   {len(val_cases)} cases, {len(val_idx):,} slices")

    return train_idx, val_idx
