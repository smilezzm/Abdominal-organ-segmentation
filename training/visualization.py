import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ipywidgets import interact, IntSlider, fixed


# Define organ labels (constant)
ORGAN_LABELS = {
    0: 'Background',
    1: 'Bladder',
    2: 'Colon',
    3: 'Femur Head',
    4: 'Kidney',
    5: 'Liver',
    6: 'Rectum',
    7: 'SmallIntestine',
    8: 'SpinalCord',
    9: 'Stomach',
    10: 'Spleen',
    11: 'Pancreas'
}


def load_and_preprocess_ct_data(ct_image_path, mask_path, target_size=(256, 256), 
                                  min_clip_value=-400, max_clip_value=400,
                                  label_rearrange=None):
    """
    Load and preprocess CT image and mask data.
    
    Args:
        ct_image_path: Path to CT image (.nii.gz)
        mask_path: Path to mask file (.nii.gz)
        target_size: Target size for resizing (default: (256, 256))
        min_clip_value: Minimum HU value for clipping (default: -400)
        max_clip_value: Maximum HU value for clipping (default: 400)
        label_rearrange: Dictionary for label remapping (optional)
    
    Returns:
        ct_data_normalized: Normalized CT data
        mask_data: Processed mask data
    """
    # Load the 3D volumes
    ct_img = nib.load(ct_image_path)
    mask_img = nib.load(mask_path)

    ct_data = ct_img.get_fdata()
    mask_data_original = mask_img.get_fdata().astype(np.int8)

    # Resize data
    current_shape = ct_data.shape
    zoom_factors = (target_size[0] / current_shape[0], target_size[1] / current_shape[1], 1.0)
    ct_data = zoom(ct_data, zoom_factors, order=1)  # order=1 for bilinear
    mask_data_original = zoom(mask_data_original, zoom_factors, order=0)  # order=0 for nearest-neighbor

    # Apply label rearrangement if provided
    if label_rearrange is not None:
        max_key = max(label_rearrange.keys())
        lut = np.array([label_rearrange[k] for k in range(max_key + 1)])
        mask_data = lut[mask_data_original]
    else:
        mask_data = mask_data_original

    # Apply clipping and normalization
    ct_data_clipped = np.clip(ct_data, min_clip_value, max_clip_value)
    ct_data_normalized = (ct_data_clipped - min_clip_value) / (max_clip_value - min_clip_value)

    print(f"CT data shape: {ct_data_normalized.shape}")
    print(f"Mask data shape: {mask_data.shape}")
    print(f"Number of slices: {ct_data_normalized.shape[2]}")
    
    return ct_data_normalized, mask_data


def generate_predictions(model, ct_data_normalized, device):
    """
    Generate predictions for all slices in a CT volume.
    
    Args:
        model: Trained PyTorch model
        ct_data_normalized: Normalized CT data (H, W, num_slices)
        device: PyTorch device (cuda or cpu)
    
    Returns:
        predicted_masks: Array of predicted masks (H, W, num_slices)
    """
    num_slices = ct_data_normalized.shape[2]
    predicted_masks = []

    print("Generating predictions for all slices...")
    with torch.no_grad():
        for z in range(num_slices):
            # Extract 2D slice
            slice_2d = ct_data_normalized[:, :, z]

            # Convert to tensor and add batch and channel dimensions
            slice_tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0).to(device)

            # Get model prediction
            output = model(slice_tensor)

            # Convert output to predicted class (argmax over class dimension)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            predicted_masks.append(pred_mask)

    predicted_masks = np.stack(predicted_masks, axis=2)
    print(f"Predictions generated. Shape: {predicted_masks.shape}")
    
    return predicted_masks


def create_colormap(num_classes):
    """
    Create a colormap for organ visualization.
    
    Args:
        num_classes: Number of organ classes
    
    Returns:
        cmap: Matplotlib colormap
        norm: Matplotlib boundary norm
    """
    # Define distinct colors for each organ class
    colors = ['black', 'red', 'blue', 'green', 'yellow', 'cyan',
              'magenta', 'orange', 'purple', 'pink', 'brown', 'gray']
    
    cmap = mcolors.ListedColormap(colors[:num_classes])
    bounds = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm


def visualize_slice(slice_idx, ct_data, gt_mask, pred_mask, cmap, norm, num_classes, organ_labels=ORGAN_LABELS):
    """
    Visualize a single slice with CT image, ground truth mask, and predicted mask.
    
    Args:
        slice_idx: Index of the slice to visualize
        ct_data: CT data volume
        gt_mask: Ground truth mask volume
        pred_mask: Predicted mask volume
        cmap: Matplotlib colormap
        norm: Matplotlib boundary norm
        num_classes: Number of classes
        organ_labels: Dictionary mapping label indices to organ names
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot CT image
    axes[0].imshow(ct_data[:, :, slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'CT Image - Slice {slice_idx + 1}/{ct_data.shape[2]}', fontsize=14)
    axes[0].axis('off')

    # Plot ground truth mask
    im1 = axes[1].imshow(gt_mask[:, :, slice_idx], cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('Ground Truth Mask', fontsize=14)
    axes[1].axis('off')

    # Plot predicted mask
    im2 = axes[2].imshow(pred_mask[:, :, slice_idx], cmap=cmap, norm=norm, interpolation='nearest')
    axes[2].set_title('Predicted Mask', fontsize=14)
    axes[2].axis('off')

    # Add a single colorbar for the predicted mask
    cbar = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Organ Class', rotation=270, labelpad=15)
    cbar.set_ticks(range(num_classes))
    cbar.set_ticklabels([organ_labels[i] for i in range(num_classes)], fontsize=8)

    # Calculate and display IoU and Dice score for this slice
    gt_slice = gt_mask[:, :, slice_idx].flatten()
    pred_slice = pred_mask[:, :, slice_idx].flatten()

    # Calculate per-class metrics
    unique_classes = np.unique(np.concatenate([gt_slice, pred_slice]))
    slice_iou = []
    slice_dice = []

    for cls in unique_classes:
        gt_cls = (gt_slice == cls)
        pred_cls = (pred_slice == cls)
        intersection = np.sum(gt_cls & pred_cls)
        union = np.sum(gt_cls | pred_cls)

        if union > 0:
            iou = intersection / union
            dice = (2 * intersection) / (np.sum(gt_cls) + np.sum(pred_cls))
        else:
            iou = 1.0  # Perfect match if both are empty
            dice = 1.0

        slice_iou.append(iou)
        slice_dice.append(dice)

    avg_iou = np.mean(slice_iou)
    avg_dice = np.mean(slice_dice)

    fig.suptitle(f'Slice IoU: {avg_iou:.4f} | Slice Dice: {avg_dice:.4f}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()


def create_interactive_visualization(ct_data_normalized, mask_data, predicted_masks, num_classes):
    """
    Create an interactive visualization with a slider to navigate through slices.
    
    Args:
        ct_data_normalized: Normalized CT data
        mask_data: Ground truth mask data
        predicted_masks: Predicted mask data
        num_classes: Number of classes
    """
    num_slices = ct_data_normalized.shape[2]
    cmap, norm = create_colormap(num_classes)
    
    # Create interactive slider
    interact(visualize_slice,
             slice_idx=IntSlider(min=0, max=num_slices-1, step=1, value=num_slices//2,
                                description='Slice Index', style={'description_width': 'initial'}),
             ct_data=fixed(ct_data_normalized),
             gt_mask=fixed(mask_data),
             pred_mask=fixed(predicted_masks),
             cmap=fixed(cmap),
             norm=fixed(norm),
             num_classes=fixed(num_classes),
             organ_labels=fixed(ORGAN_LABELS))


def calculate_and_visualize_dice_scores(mask_data, predicted_masks, num_classes, organ_labels=ORGAN_LABELS):
    """
    Calculate overall Dice scores for each label and visualize them.
    
    Args:
        mask_data: Ground truth mask data
        predicted_masks: Predicted mask data
        num_classes: Number of classes
        organ_labels: Dictionary mapping label indices to organ names
    
    Returns:
        label_dice_scores: Dictionary of dice scores per label
        mean_dice: Mean dice score across all labels
    """
    print("Calculating overall Dice scores for each label...")

    # Flatten the entire 3D volumes for computation
    gt_flat = mask_data.flatten()
    pred_flat = predicted_masks.flatten()

    # Get unique labels present in ground truth (excluding background label 0)
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background

    # Dictionary to store dice scores for each label
    label_dice_scores = {}

    print(f"\nDice Scores by Organ Class:")
    print("-" * 60)

    for label in unique_labels:
        # Create binary masks for this specific label
        gt_binary = (gt_flat == label)
        pred_binary = (pred_flat == label)
        
        # Calculate intersection and union
        intersection = np.sum(gt_binary & pred_binary)
        gt_sum = np.sum(gt_binary)
        pred_sum = np.sum(pred_binary)
        
        # Calculate Dice score
        if gt_sum + pred_sum > 0:
            dice = (2 * intersection) / (gt_sum + pred_sum)
        else:
            dice = 1.0  # Perfect match if both are empty
        
        label_dice_scores[label] = dice
        
        # Get organ name from label dictionary
        organ_name = organ_labels.get(int(label), f"Unknown_{label}")
        print(f"Label {int(label):2d} ({organ_name:15s}): Dice = {dice:.4f}")

    # Calculate mean Dice score across all non-background labels
    mean_dice = np.mean(list(label_dice_scores.values()))
    print("-" * 60)
    print(f"Mean Dice Score (excluding background): {mean_dice:.4f}")

    # Visualize Dice scores
    fig, ax = plt.subplots(figsize=(12, 6))
    labels_list = list(label_dice_scores.keys())
    dice_values = list(label_dice_scores.values())
    organ_names = [organ_labels.get(int(label), f"Label_{label}") for label in labels_list]

    bars = ax.bar(range(len(labels_list)), dice_values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels([f"{int(l)}: {n}" for l, n in zip(labels_list, organ_names)], 
                        rotation=45, ha='right')
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_xlabel('Organ Class', fontsize=12)
    ax.set_title('Dice Score by Organ Class (Excluding Background)', fontsize=14, fontweight='bold')
    ax.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend()

    # Add value labels on top of bars
    for i, (bar, dice) in enumerate(zip(bars, dice_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{dice:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    return label_dice_scores, mean_dice
