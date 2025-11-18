# slicing_v2.py
import os
import argparse
import cv2
import json
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--kits_path', type=str, required=True,
                    help='Path to KiTS19 original data (case_xxxx folders)')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Output directory for preprocessed .npy files')
parser.add_argument('--task', type=str, default='tumor', choices=['tumor', 'kidney', 'both'],
                    help='Type of segmentation (binary foreground)')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--hu_min', type=int, default=-200)
parser.add_argument('--hu_max', type=int, default=250)

args = parser.parse_args()


# ---------------------------------------------------------
# 1. HU normalization
# ---------------------------------------------------------
def normalize_hu(img, hu_min=-200, hu_max=250):
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min)  # 0~1 normalize
    return img.astype(np.float32)


# ---------------------------------------------------------
# 2. Create One-hot mask (2 channels)
# ---------------------------------------------------------
def create_mask_2ch(mask_slice, task='tumor'):
    """
    KiTS mask values:
      0 = background
      1 = kidney
      2 = tumor
    """
    if task == 'tumor':
        fg = (mask_slice == 2).astype(np.uint8)
    elif task == 'kidney':
        fg = (mask_slice > 0).astype(np.uint8)
    elif task == 'both':   # kidney & tumor 동일 foreground
        fg = (mask_slice > 0).astype(np.uint8)
    else:
        raise ValueError("Invalid task")

    bg = 1 - fg
    mask_2ch = np.stack([bg, fg], axis=0)  # (2,H,W)
    return mask_2ch


# ---------------------------------------------------------
# 3. Read KiTS case
# ---------------------------------------------------------
def read_kits_case(case_dir):
    img_path = glob(os.path.join(case_dir, "imaging*.nii.gz"))[0]
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")

    vol = nib.load(img_path).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.int16)

    return vol, seg


# ---------------------------------------------------------
# 4. Main slicing loop
# ---------------------------------------------------------
def main():
    os.makedirs(args.out_dir, exist_ok=True)

    case_dirs = sorted(glob(os.path.join(args.kits_path, "case_*")))

    images = []
    masks = []
    case_ids = []
    slice_ids = []

    print(f"[+] Found {len(case_dirs)} cases")
    print("[*] Processing...")

    for case_dir in tqdm(case_dirs):
        case_name = os.path.basename(case_dir)  # case_00000
        case_id = case_name.split('_')[-1]      # "00000"

        vol, seg = read_kits_case(case_dir)
        depth = vol.shape[0]

        for i in range(depth):
            ct_slice = vol[i, :, :]
            mask_slice = seg[i, :, :]

            # Resize (both image and mask)
            ct_slice = cv2.resize(ct_slice, (args.img_size, args.img_size),
                                  interpolation=cv2.INTER_LINEAR)
            mask_slice = cv2.resize(mask_slice, (args.img_size, args.img_size),
                                    interpolation=cv2.INTER_NEAREST)

            # HU normalize
            ct_slice = normalize_hu(ct_slice, hu_min=args.hu_min, hu_max=args.hu_max)

            # Expand image channel -> (1, H, W)
            ct_slice = ct_slice[np.newaxis, :, :]

            # Create One-hot mask (2,H,W)
            mask_2ch = create_mask_2ch(mask_slice, task=args.task)

            images.append(ct_slice)
            masks.append(mask_2ch)
            case_ids.append(int(case_id))
            slice_ids.append(i)

    # Stack arrays
    images = np.stack(images, axis=0)  # (N,1,H,W)
    masks = np.stack(masks, axis=0)    # (N,2,H,W)
    case_ids = np.array(case_ids)
    slice_ids = np.array(slice_ids)

    print("[*] Saving .npy files...")
    np.save(os.path.join(args.out_dir, "images.npy"), images)
    np.save(os.path.join(args.out_dir, "masks.npy"), masks)
    np.save(os.path.join(args.out_dir, "case_ids.npy"), case_ids)
    np.save(os.path.join(args.out_dir, "slice_ids.npy"), slice_ids)

    print("\n[✓] Done saving .npy files!")
    print(f"Saved images: {images.shape}")
    print(f"Saved masks:  {masks.shape}")
    print(f"Saved case_ids: {case_ids.shape}")
    print(f"Saved slice_ids: {slice_ids.shape}")

    # ---------------------------------------------------------
    # 5. Create case_mapping.pkl (NEW)
    # ---------------------------------------------------------
    print("\n[*] Creating case_mapping.pkl ...")

    case_mapping = {}
    unique_cases = np.unique(case_ids)

    for cid in unique_cases:
        indices = np.where(case_ids == cid)[0].tolist()
        case_mapping[str(cid).zfill(5)] = {
            "indices": indices
        }

    import pickle
    with open(os.path.join(args.out_dir, "case_mapping.pkl"), "wb") as f:
        pickle.dump(case_mapping, f)

    print("[✓] Saved case_mapping.pkl")
    print(f"Total cases: {len(case_mapping)}")



if __name__ == "__main__":
    main()
