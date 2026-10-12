"""
PURPOSE: Containing the data loader for loading and preprocessing your data, including leakagefree train/validation/test splitting functions and data augmentations.

Data Pipeline & Alignment: Enforce strict patient-level splitting. Standardize intensity normalization and one-hot multi-channel mask encoding. Preserve exact spatial alignment between images and ground truth masks (see Appendix B for Nifti loading).

The classes for the mask are: [0, 1, 2, 3, 4, 5]
"""

from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def extract_key(name: str):
    """
    Goes from 'case_004_week_0_slice_3.nii.gz' -> (4, 0, 3)
    """
    _, patient_num, _, week_num, _, slice_part = name.split("_")
    slice_num, _, _ = slice_part.split(".")

    return (int(patient_num), int(week_num), int(slice_num))

def build_img_seg_pairs(img_dir, seg_dir):
    """
    Returns a dictionary that uses (patient, week, slice) as the key, and the value being a tuple (image, segment).
    """

    img_files = sorted(os.listdir(img_dir), key=extract_key)
    seg_files = sorted(os.listdir(seg_dir), key=extract_key)

    # Should be same, as we're relying on indices to pair them up
    assert len(img_files) == len(seg_files), "not same file count between images and masks" 

    img_seg_pairs = {}
    for i in range(len(img_files)):
        full_img_path = os.path.join(img_dir, img_files[i])
        full_seg_path = os.path.join(seg_dir, seg_files[i])

        img_seg_pairs[extract_key(img_files[i])] =  (full_img_path, full_seg_path)

    return img_seg_pairs

def get_seg_classes(seg_dir):
    labelIDs = set()

    for filename in os.listdir(seg_dir):
        full_path = os.path.join(seg_dir, filename)
        seg = nib.load(full_path).get_fdata(caching='unchanged') # A grid of numbers, with entries as the labels IDs to segment
        arr = np.rint(seg).astype(int) # Same grid, but with clean integers

        labelIDs.update(np.unique(arr).tolist()) # Add all unique labels to set

    return len(labelIDs)


class HipMRIDataset2D(Dataset):
    def __init__(self, pairs, num_classes):
        self.pairs = pairs
        self.keys = list(pairs.keys()) # Already in order/sorted thanks to dictionary's order
        self.num_classes = num_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, seg_path = self.pairs[self.keys[i]]

        # Load MRI Slice
        img = nib.load(img_path).get_fdata(caching='unchanged') # (256, 128)
        img = img.astype(np.float32) # PyTorch expects float32, not float64

        # z-score normalisation for the image
        img_mean = img.mean()
        img_std = img.std() + 1e-8 # "+ 1e-8" To avoid divide by 0 error
        img = (img - img_mean) / img_std

        # Convert to tensor (add channel/class axis)
        img_tensor = torch.from_numpy(img).unsqueeze(0) # (256, 128) -> (1, 256, 128)



        # Load segmentation label
        seg = nib.load(seg_path).get_fdata(caching='unchanged') # (256, 128)
        seg = seg.astype(int)
        seg_tensor = torch.from_numpy(seg)

        # One-hot encoding on the segmentation mask
        seg_onehot_tensor = F.one_hot(seg_tensor, num_classes=self.num_classes) # (256, 128, 6)
        seg_onehot_tensor = seg_onehot_tensor.permute(2, 0, 1).float() # (256, 128, 6) -> (6, 256, 128)


        # Key, if needed while in the loops of the dataset
        key_tensor = torch.tensor(self.keys[i])


        return img_tensor, seg_onehot_tensor, key_tensor 









if __name__ == "__main__":
    IMAGE_DIR = r"Dataset\keras_slices_data\keras_slices_validate"
    SEG_DIR = r"Dataset\keras_slices_data\keras_slices_seg_validate"

    pairs = build_img_seg_pairs(IMAGE_DIR, SEG_DIR)
    num_classes = get_seg_classes(SEG_DIR)

    train_dataset = HipMRIDataset2D(pairs, num_classes)
    img, seg, key = train_dataset[0]