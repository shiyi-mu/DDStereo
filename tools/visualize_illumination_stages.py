import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def apply_gamma(img_np, gamma):
    """Apply gamma correction."""
    img_float = img_np.astype(np.float32) / 255.0
    img_out = np.power(img_float, gamma)
    img_out = np.clip(img_out * 255.0, 0, 255).astype(np.uint8)
    return img_out


def apply_brightness(img_np, factor):
    """Apply brightness scaling."""
    img_out = img_np.astype(np.float32) * factor
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    return img_out


def visualize_stages(config_path, output_dir, sample_indices=None,
                     gamma_values=None, brightness_values=None,
                     adjust_mode='gamma', num_samples=4):
    """Visualize input images at different illumination stages."""

    os.makedirs(output_dir, exist_ok=True)

    # Load config to get image paths
    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    dataset_cfg = cfg['dataset']

    # Determine val subset and paths
    val_subset = dataset_cfg.get('val_subset', 'ori_val')
    if val_subset == 'ood_val':
        root_dir = dataset_cfg['root_dir_ood_val']
        eval_txt = dataset_cfg['eval_txt_ood_val']
    else:
        root_dir = dataset_cfg['root_dir_ori_val']
        eval_txt = dataset_cfg['eval_txt_ori_val']

    image_dir = os.path.join(root_dir, 'image_2')
    image3_dir = os.path.join(root_dir, 'image_3')

    # Load image list
    idx_list = [x.strip() for x in open(eval_txt).readlines()]

    if sample_indices is None:
        # Randomly sample images
        np.random.seed(42)
        sample_indices = np.random.choice(len(idx_list), min(num_samples, len(idx_list)), replace=False)

    # Determine test values
    if adjust_mode == 'gamma':
        # gamma > 1 = darker, default range 1.0 -> 5.0
        test_values = gamma_values if gamma_values is not None else [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    elif adjust_mode == 'brightness':
        # brightness < 1 = darker, default range 0.1 -> 1.0 (dark to bright)
        test_values = brightness_values if brightness_values is not None else [0.1, 0.4, 1.0]
    else:
        test_values = gamma_values if gamma_values is not None else [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]

    print(f"Visualizing {len(sample_indices)} samples with {len(test_values)} {adjust_mode} stages...")

    for samp_idx in sample_indices:
        img_id = idx_list[samp_idx]

        # Load left image only
        img_path = os.path.join(image_dir, f'{img_id}.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, f'{img_id}.jpg')

        if not os.path.exists(img_path):
            print(f"Skip {img_id}: image not found")
            continue

        img_orig = np.array(Image.open(img_path))
        img_h, img_w = img_orig.shape[:2]
        aspect = img_w / img_h  # image width / height ratio

        # Create figure: 1 row x 6 cols grid, tight layout
        # figsize matches image aspect ratio so no stretching or padding
        nrows = 1
        ncols = 3
        row_h = 2.0
        fig_w = ncols * aspect * row_h
        fig_h = nrows * row_h

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = axes.flatten()

        for idx, val in enumerate(test_values):
            if adjust_mode == 'gamma':
                img_left = apply_gamma(img_orig, val)
                label = f"γ={val:.1f}"
            elif adjust_mode == 'brightness':
                img_left = apply_brightness(img_orig, val)
                label = f"B={val:.1f}"
            else:
                img_left = apply_brightness(apply_gamma(img_orig, val), val)
                label = f"γ={val:.1f} B={val:.1f}"

            axes[idx].imshow(img_left)
            axes[idx].axis('off')

            # Label on the left side of each subplot
            axes[idx].annotate(label, xy=(0.01, 0.5), xycoords='axes fraction',
                              fontsize=18, color='yellow', fontweight='bold',
                              ha='left', va='center')

        # No gaps between subplots, no margins at all
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)

        out_path = os.path.join(output_dir, f'{adjust_mode}_sample_{img_id}.png')
        plt.savefig(out_path, dpi=150, bbox_inches=None, pad_inches=0)
        plt.close()
        print(f"Saved: {out_path}")

    # Create a summary comparison figure with histograms
    print("\nGenerating histogram analysis...")
    create_histogram_analysis(idx_list, image_dir, test_values, adjust_mode, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


def create_histogram_analysis(idx_list, image_dir, test_values, adjust_mode, output_dir):
    """Create histogram analysis of pixel distributions at different stages."""

    # Sample a few images for histogram
    np.random.seed(42)
    hist_samples = np.random.choice(len(idx_list), min(100, len(idx_list)), replace=False)

    fig, axes = plt.subplots(2, len(test_values), figsize=(2.2 * len(test_values), 6))

    for col, val in enumerate(test_values):
        all_pixels = []

        for samp_idx in hist_samples:
            img_id = idx_list[samp_idx]
            img_path = os.path.join(image_dir, f'{img_id}.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(image_dir, f'{img_id}.jpg')
            if not os.path.exists(img_path):
                continue

            img = np.array(Image.open(img_path)).astype(np.float32)

            if adjust_mode == 'gamma':
                img = apply_gamma(img, val)
            elif adjust_mode == 'brightness':
                img = apply_brightness(img, val)
            else:
                img = apply_brightness(apply_gamma(img, val), val)

            all_pixels.extend(img.flatten())

        all_pixels = np.array(all_pixels)

        # Top: histogram
        axes[0, col].hist(all_pixels, bins=50, range=(0, 255), color='steelblue', edgecolor='white', alpha=0.7)
        if adjust_mode == 'gamma':
            axes[0, col].set_title(f'γ={val:.1f}', fontsize=10)
        else:
            axes[0, col].set_title(f'B={val:.1f}', fontsize=10)
        axes[0, col].set_xlim(0, 255)
        if col == 0:
            axes[0, col].set_ylabel('Pixel Count', fontsize=10)
        axes[0, col].tick_params(labelsize=8)

        # Bottom: mean/percentile bars
        mean_val = np.mean(all_pixels)
        p25 = np.percentile(all_pixels, 25)
        p50 = np.percentile(all_pixels, 50)
        p75 = np.percentile(all_pixels, 75)

        bar_data = [p25, p50, p75, mean_val]
        bar_labels = ['P25', 'P50', 'P75', 'Mean']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        bars = axes[1, col].bar(range(4), bar_data, color=colors, edgecolor='white')
        axes[1, col].set_xticks(range(4))
        axes[1, col].set_xticklabels(bar_labels, fontsize=8, rotation=45)
        axes[1, col].set_ylim(0, 255)
        if col == 0:
            axes[1, col].set_ylabel('Pixel Value', fontsize=10)
        axes[1, col].tick_params(labelsize=8)

        # Add value labels on bars
        for bar, v in zip(bars, bar_data):
            axes[1, col].text(bar.get_x() + bar.get_width()/2., v + 3,
                             f'{v:.0f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'Pixel Distribution Analysis ({adjust_mode}) - 100 random samples', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, f'{adjust_mode}_histogram_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize illumination stages')
    parser.add_argument('--config', type=str, default='configs/101-OoD-arind-train-ood-val.yaml',
                        help='Config file path')
    parser.add_argument('--output_dir', type=str, default='outputs/illumination_visualization',
                        help='Output directory')
    parser.add_argument('--adjust_mode', type=str, default='brightness', choices=['gamma', 'brightness', 'both'])
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--sample_ids', type=str, default=None,
                        help='Specific image IDs to visualize (comma-separated)')
    args = parser.parse_args()

    sample_indices = None
    if args.sample_ids:
        sample_indices = [int(x.strip()) for x in args.sample_ids.split(',')]

    visualize_stages(
        config_path=args.config,
        output_dir=args.output_dir,
        sample_indices=sample_indices,
        adjust_mode=args.adjust_mode,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
