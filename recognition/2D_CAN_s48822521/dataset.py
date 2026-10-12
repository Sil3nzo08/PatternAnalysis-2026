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


if __name__ == "__main__":
    IMAGE_DIR = r"Dataset\keras_slices_data\keras_slices_validate"
    SEG_DIR = r"Dataset\keras_slices_data\keras_slices_seg_validate"

    pairs = build_img_seg_pairs(IMAGE_DIR, SEG_DIR)
    print(pairs)