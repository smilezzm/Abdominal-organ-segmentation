"""
Dataset Classes and Utilities for Medical Image Segmentation

This module contains PyTorch Dataset classes and utility functions for loading
preprocessed 2D medical image slices with corresponding masks and distance maps.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class NPYDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed .npy files (images, masks, distance maps).
    
    Args:
        file_paths (list): List of tuples (image_path, mask_path, distance_map_path)
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path, mask_path, distance_map_path = self.file_paths[idx]

        # Load image and mask data
        image = np.load(image_path)
        mask = np.load(mask_path)
        distance_map = np.load(distance_map_path)

        # Add a channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        distance_map = np.expand_dims(distance_map, axis=0)

        # Convert to PyTorch tensors
        # Image should be float32 for model input
        image_tensor = torch.from_numpy(image).float()
        # Mask should be long for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask).long()
        distance_map_tensor = torch.from_numpy(distance_map).float()

        return image_tensor, mask_tensor, distance_map_tensor


def load_and_pair_files(output_base_path):
    """
    Load and pair image, mask, and distance map files from preprocessed directories.
    
    Args:
        output_base_path (str): Base path containing images_2d, masks_2d, and distance_maps_2d subdirectories
        
    Returns:
        list: List of tuples (image_path, mask_path, distance_map_path) for all paired files
    """
    output_images_path = os.path.join(output_base_path, 'images_2d/')
    output_masks_path = os.path.join(output_base_path, 'masks_2d/')
    output_distance_maps_path = os.path.join(output_base_path, 'distance_maps_2d/')

    # Get all .npy files
    image_files = sorted([os.path.join(output_images_path, f) 
                         for f in os.listdir(output_images_path) if f.endswith('.npy')])
    mask_files = sorted([os.path.join(output_masks_path, f) 
                        for f in os.listdir(output_masks_path) if f.endswith('.npy')])
    distance_map_files = sorted([os.path.join(output_distance_maps_path, f) 
                                for f in os.listdir(output_distance_maps_path) if f.endswith('.npy')])

    # Create dictionaries for efficient lookup and pairing
    mask_dict = {os.path.basename(f): f for f in mask_files}
    distance_map_dict = {os.path.basename(f): f for f in distance_map_files}

    # Pair files based on matching filenames
    paired_file_paths = []
    for img_path in image_files:
        img_filename = os.path.basename(img_path)
        if img_filename in mask_dict and img_filename in distance_map_dict:
            paired_file_paths.append((img_path, mask_dict[img_filename], distance_map_dict[img_filename]))

    print(f"Found {len(paired_file_paths)} paired (image+mask+distance_map) files.")
    return paired_file_paths


def create_train_val_datasets(paired_file_paths, test_size=0.2, random_state=42):
    """
    Split paired files into training and validation datasets.
    
    Args:
        paired_file_paths (list): List of tuples (image_path, mask_path, distance_map_path)
        test_size (float): Fraction of data to use for validation (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_dataset, val_dataset) - NPYDataset instances for training and validation
    """
    # Split the paired file paths into training and validation sets
    train_files, val_files = train_test_split(paired_file_paths, test_size=test_size, random_state=random_state)

    # Create dataset instances
    train_dataset = NPYDataset(train_files)
    val_dataset = NPYDataset(val_files)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset
