"""
Medical Image Preprocessing Functions

This module contains functions for preprocessing 3D medical image data (NIfTI format)
into 2D slices with augmentation for training deep learning models.
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, distance_transform_edt


def random_brightness_contrast(image):
    """
    Applies random brightness and contrast adjustments to an image.
    Image pixel values are assumed to be normalized between 0 and 1.
    
    Args:
        image (np.ndarray): Input image with values in [0, 1]
        
    Returns:
        np.ndarray: Image with random brightness/contrast applied
    """
    # Random brightness factor between 0.97 and 1.03
    brightness_factor = np.random.uniform(0.97, 1.03)
    # Random contrast factor between 0.96 and 1.04
    contrast_factor = np.random.uniform(0.96, 1.04)

    # Apply contrast adjustment first
    image = image * contrast_factor
    # Apply brightness adjustment. Adjusting by (brightness_factor - 1) ensures 1 means no change.
    image = image + (brightness_factor - 1)

    # Clip image pixel values to stay within the valid range [0, 1]
    image = np.clip(image, 0.0, 1.0)

    return image


def preprocess_dataset(image_input_path, mask_input_path, output_base_path, 
                       label_rearrange_dict, label_frequency=None,
                       min_clip_value=-400, max_clip_value=400, 
                       target_size=(256, 256), num_brightness_variations=4):
    """
    Preprocess 3D medical images into 2D slices with augmentation.
    
    Args:
        image_input_path (str): Path to directory containing CT image .nii.gz files
        mask_input_path (str): Path to directory containing mask .nii.gz files
        output_base_path (str): Base path for output directories
        label_rearrange_dict (dict): Dictionary mapping original labels to new labels
        label_frequency (dict, optional): Dictionary to accumulate label frequencies
        min_clip_value (float): Minimum HU value for clipping
        max_clip_value (float): Maximum HU value for clipping
        target_size (tuple): Target size for resized slices (height, width)
        num_brightness_variations (int): Number of brightness/contrast variations (including original)
        
    Returns:
        dict: Updated label_frequency dictionary
    """
    # Define subdirectories for images, masks, and distance maps
    output_images_path = os.path.join(output_base_path, 'images_2d/')
    output_masks_path = os.path.join(output_base_path, 'masks_2d/')
    output_distance_maps_path = os.path.join(output_base_path, 'distance_maps_2d/')

    # Create the directories if they do not already exist
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)
    os.makedirs(output_distance_maps_path, exist_ok=True)

    # Initialize label frequency if not provided
    if label_frequency is None:
        label_frequency = {i: 0 for i in range(12)}

    # Get sorted lists of all .nii.gz files from both directories
    image_files_raw = sorted([f for f in os.listdir(image_input_path) if f.endswith('.nii.gz')])
    mask_files_raw = sorted([f for f in os.listdir(mask_input_path) if f.endswith('.nii.gz')])

    # Create dictionaries for easy lookup
    # Handle different naming conventions (with or without '_0000' suffix)
    image_dict = {}
    for f in image_files_raw:
        base = f.replace('.nii.gz', '').replace('_0000', '')
        image_dict[base] = f
    
    mask_dict = {}
    for f in mask_files_raw:
        base = f.replace('.nii.gz', '')
        mask_dict[base] = f

    # Pair each image file with its corresponding mask file
    paired_files = []
    for base_name in sorted(image_dict.keys()):
        if base_name in mask_dict:
            image_full_path = os.path.join(image_input_path, image_dict[base_name])
            mask_full_path = os.path.join(mask_input_path, mask_dict[base_name])
            paired_files.append((image_full_path, mask_full_path))

    print(f"Starting processing of {len(paired_files)} image-mask pairs...")
    print(f"  Output path: {output_base_path}")
    print(f"  Target size: {target_size}")
    print(f"  HU clip range: [{min_clip_value}, {max_clip_value}]")
    print(f"  Brightness variations: {num_brightness_variations}")

    processed_count = 0

    for image_full_path, mask_full_path in paired_files:
        # Extract base name for consistent file naming
        base_name = os.path.basename(image_full_path).replace('.nii.gz', '').replace('_0000', '')

        try:
            # Load 3D image and mask data into memory
            image_nifti = nib.load(image_full_path)
            mask_nifti = nib.load(mask_full_path)

            image_data = image_nifti.get_fdata()
            # Ensure mask data is integer type for labels, using np.int8 for memory efficiency
            original_mask_data = mask_nifti.get_fdata().astype(np.int8)
            
            # Rearrange labels using lookup table
            max_key = max(label_rearrange_dict.keys())
            lut = np.array([label_rearrange_dict[k] for k in range(max_key + 1)])
            mask_data = lut[original_mask_data]

            # Count the frequency of labels
            unique, counts = np.unique(mask_data, return_counts=True)
            if len(unique) != 1:
                for u, c in zip(unique, counts):
                    if u in label_frequency:
                        label_frequency[u] += c

            # Basic shape consistency check
            if image_data.shape != mask_data.shape:
                print(f"Warning: Shape mismatch for {base_name}. Skipping. Image: {image_data.shape}, Mask: {mask_data.shape}")
                del image_nifti, mask_nifti, image_data, mask_data
                continue

            num_slices = image_data.shape[2]  # Number of slices along the depth (z) dimension

            for z in range(num_slices):
                # Extract 2D slices for current depth (z)
                original_image_slice = image_data[:, :, z]
                original_mask_slice = mask_data[:, :, z]

                # --- Resizing slices to target_size ---
                current_shape = original_image_slice.shape
                zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1])

                # Resize image data (bilinear interpolation for continuous data)
                resized_image_slice = zoom(original_image_slice, zoom_factors, order=1)

                # Resize mask data (nearest-neighbor interpolation for discrete labels)
                resized_mask_slice = zoom(original_mask_slice, zoom_factors, order=0)

                # --- Image Preprocessing: Clip and Rescale ---
                clipped_image_slice = np.clip(resized_image_slice, min_clip_value, max_clip_value)
                preprocessed_image_slice = (clipped_image_slice - min_clip_value) / (max_clip_value - min_clip_value)

                # --- Calculate distance maps ---
                binary_slice = (resized_mask_slice > 0).astype(np.uint8)
                distance_map_slice = distance_transform_edt(binary_slice) + distance_transform_edt(1 - binary_slice)
                
                # If unannotated, set distance=large so that this slice doesn't matter
                if np.count_nonzero(resized_mask_slice) == 0:
                    distance_map_slice = np.ones_like(resized_mask_slice, dtype=np.float32) * resized_mask_slice.shape[0]

                # --- Data Augmentation: Brightness/Contrast and Rotations ---
                intensity_variations = []
                # Original brightness/contrast variation (bc_var0)
                intensity_variations.append((preprocessed_image_slice, "bc_var0"))
                
                # Additional random brightness/contrast variations
                for i in range(1, num_brightness_variations):
                    bc_image_slice = random_brightness_contrast(preprocessed_image_slice.copy())
                    intensity_variations.append((bc_image_slice, f"bc_var{i}"))

                for img_var, bc_tag in intensity_variations:
                    slices_to_save = []
                    # Add original rotation (0 degrees)
                    slices_to_save.append((img_var, resized_mask_slice, distance_map_slice, 0))

                    # --- Rotations for each brightness/contrast variation ---
                    for k, angle in enumerate([90, 180, 270]):
                        rotated_image_slice = np.rot90(img_var, k=k + 1)
                        rotated_mask_slice = np.rot90(resized_mask_slice, k=k + 1)
                        rotated_distance_maps_slice = np.rot90(distance_map_slice, k=k + 1)
                        slices_to_save.append((rotated_image_slice, rotated_mask_slice, rotated_distance_maps_slice, angle))

                    # --- Save all generated slices ---
                    for img_slice, msk_slice, dist_map_slice, angle in slices_to_save:
                        # Construct filenames including original base name, slice index, bc tag, and rotation angle
                        image_filename = f"{base_name}_slice_{z:03d}_{bc_tag}_rot_{angle}.npy"
                        mask_filename = f"{base_name}_slice_{z:03d}_{bc_tag}_rot_{angle}.npy"
                        distance_map_filename = f"{base_name}_slice_{z:03d}_{bc_tag}_rot_{angle}.npy"

                        # Save image slices as float32 to preserve intensity information
                        np.save(os.path.join(output_images_path, image_filename), img_slice.astype(np.float32))
                        # Save mask slices as int8 to keep them as labels and be memory efficient
                        np.save(os.path.join(output_masks_path, mask_filename), msk_slice.astype(np.int8))
                        # Save distance map slices as float32
                        np.save(os.path.join(output_distance_maps_path, distance_map_filename), dist_map_slice.astype(np.float32))

            processed_count += 1
            variations_per_slice = num_brightness_variations * 4  # 4 rotations per brightness variation
            print(f"Processed {processed_count}/{len(paired_files)}: {base_name} ({num_slices} slices, {variations_per_slice} variations each)")

            # Explicitly clear 3D data from memory for the current pair to avoid memory overflow
            del image_nifti, mask_nifti, image_data, mask_data

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            # Attempt to clear memory even on error to prevent issues for subsequent files
            if 'image_nifti' in locals():
                del image_nifti
            if 'mask_nifti' in locals():
                del mask_nifti
            if 'image_data' in locals():
                del image_data
            if 'mask_data' in locals():
                del mask_data
            continue

    print(f"\nFinished processing {processed_count} files.")
    return label_frequency


def preprocess_pku_dataset(image_input_path, mask_input_path, output_base_path, 
                           label_frequency=None, **kwargs):
    """
    Preprocess PKU dataset with label rearrangement.
    
    Args:
        image_input_path (str): Path to PKU CT images
        mask_input_path (str): Path to PKU masks
        output_base_path (str): Base output path
        label_frequency (dict, optional): Label frequency dictionary
        **kwargs: Additional arguments passed to preprocess_dataset
        
    Returns:
        dict: Updated label_frequency dictionary
    """
    # PKU label rearrangement
    pku_label_rearrange = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9}
    
    print("=" * 60)
    print("Processing PKU Dataset")
    print("=" * 60)
    
    return preprocess_dataset(
        image_input_path=image_input_path,
        mask_input_path=mask_input_path,
        output_base_path=output_base_path,
        label_rearrange_dict=pku_label_rearrange,
        label_frequency=label_frequency,
        **kwargs
    )


def preprocess_other_dataset(image_input_path, mask_input_path, output_base_path,
                             label_frequency=None, **kwargs):
    """
    Preprocess 'Other' (AbdomenCT-1K) dataset with label rearrangement.
    
    Args:
        image_input_path (str): Path to Other dataset CT images
        mask_input_path (str): Path to Other dataset masks
        output_base_path (str): Base output path
        label_frequency (dict, optional): Label frequency dictionary
        **kwargs: Additional arguments passed to preprocess_dataset
        
    Returns:
        dict: Updated label_frequency dictionary
    """
    # Other dataset label rearrangement (adds Spleen=10 and Pancreas=11)
    other_label_rearrange = {0: 0, 1: 5, 2: 4, 3: 10, 4: 11}
    
    print("=" * 60)
    print("Processing Other (AbdomenCT-1K) Dataset")
    print("=" * 60)
    
    return preprocess_dataset(
        image_input_path=image_input_path,
        mask_input_path=mask_input_path,
        output_base_path=output_base_path,
        label_rearrange_dict=other_label_rearrange,
        label_frequency=label_frequency,
        **kwargs
    )

