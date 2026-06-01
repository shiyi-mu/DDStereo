import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import matplotlib.pyplot as plt

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.save_helper import load_checkpoint
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


class IlluminationDataset(KITTI_Dataset):
    """KITTI Dataset with illumination (gamma/brightness) adjustment."""

    def __init__(self, split, cfg, subset_train="ori_train", subset_val="ori_val",
                 gamma=1.0, brightness_factor=1.0, adjust_mode='gamma'):
        super().__init__(split, cfg, subset_train, subset_val)
        self.gamma = gamma
        self.brightness_factor = brightness_factor
        self.adjust_mode = adjust_mode
        print(f">>>IlluminationTest>>> gamma={gamma}, brightness={brightness_factor}, mode={adjust_mode}")

    def _adjust_image(self, img):
        """Apply illumination adjustment to PIL Image."""
        img_np = np.array(img).astype(np.float32) / 255.0

        if self.adjust_mode == 'gamma':
            # Gamma correction: I_out = I_in ^ gamma
            # gamma < 1: brighter (dark pixels boosted), gamma > 1: darker
            img_np = np.power(img_np, self.gamma)
        elif self.adjust_mode == 'brightness':
            # Simple brightness scaling
            img_np = img_np * self.brightness_factor
        elif self.adjust_mode == 'both':
            img_np = np.power(img_np, self.gamma) * self.brightness_factor

        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def get_image(self, idx):
        img = super().get_image(idx)
        return self._adjust_image(img)

    def get_image3(self, idx):
        img3 = super().get_image3(idx)
        return self._adjust_image(img3)


def build_illumination_dataloader(cfg, gamma=1.0, brightness_factor=1.0, adjust_mode='gamma'):
    """Build dataloader with illumination adjustment."""
    from torch.utils.data import DataLoader

    # Only support val/test split for robustness testing
    dataset = IlluminationDataset(
        split=cfg['test_split'],
        cfg=cfg,
        subset_train=cfg['train_subsets'][0] if 'train_subsets' in cfg else 'ori_train',
        subset_val=cfg.get('val_subset', 'ori_val'),
        gamma=gamma,
        brightness_factor=brightness_factor,
        adjust_mode=adjust_mode
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    return test_loader


def parse_eval_log(log_file):
    """Parse evaluation log to extract AP metrics."""
    results = {}
    current_category = None

    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Car AP' in line:
                current_category = 'Car'
            elif 'Pedestrian AP' in line:
                current_category = 'Pedestrian'
            elif 'Cyclist AP' in line:
                current_category = 'Cyclist'
            elif 'OoD AP' in line:
                current_category = 'OoD'
            elif current_category and '3d ' in line and 'AP:' in line:
                # Parse 3D AP line: "3d   AP:90.50, 87.63, 79.40"
                parts = line.split('AP:')
                if len(parts) == 2:
                    values = parts[1].split(',')
                    if len(values) == 3:
                        easy = float(values[0].strip())
                        moderate = float(values[1].strip())
                        hard = float(values[2].strip())
                        results[current_category] = {
                            'easy': easy,
                            'moderate': moderate,
                            'hard': hard
                        }
                        current_category = None
    return results


def plot_results(all_results, output_dir, adjust_mode):
    """Plot illumination robustness curves."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    gammas = [r['gamma'] for r in all_results]
    brightness_factors = [r['brightness'] for r in all_results]

    categories = ['Car', 'Pedestrian', 'Cyclist', 'OoD']
    difficulties = ['easy', 'moderate', 'hard']
    difficulty_labels = {'easy': 'Easy', 'moderate': 'Moderate', 'hard': 'Hard'}

    x_label = 'Gamma' if adjust_mode == 'gamma' else 'Brightness Factor'
    x_values = gammas if adjust_mode == 'gamma' else brightness_factors

    # Plot 1: All categories, moderate difficulty
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Illumination Robustness Test ({adjust_mode})', fontsize=16)

    for idx, cat in enumerate(categories):
        ax = axes[idx // 2, idx % 2]
        for diff in difficulties:
            y_values = []
            for r in all_results:
                if cat in r['metrics'] and diff in r['metrics'][cat]:
                    y_values.append(r['metrics'][cat][diff])
                else:
                    y_values.append(0)
            ax.plot(x_values, y_values, marker='o', label=difficulty_labels[diff], linewidth=2)

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('AP (%)', fontsize=12)
        ax.set_title(f'{cat} 3D Detection AP', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'illumination_robustness_{adjust_mode}.png'), dpi=150)
    plt.close()

    # Plot 2: Moderate difficulty for all categories in one figure
    fig, ax = plt.subplots(figsize=(14, 9))
    for cat in categories:
        y_values = []
        for r in all_results:
            if cat in r['metrics'] and 'moderate' in r['metrics'][cat]:
                y_values.append(r['metrics'][cat]['moderate'])
            else:
                y_values.append(0)
        ax.plot(x_values, y_values, marker='o', markersize=10, label=cat, linewidth=3)

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel('AP (%)', fontsize=24)
    ax.set_title(f'Illumination Robustness - Moderate Difficulty ({adjust_mode})', fontsize=26)
    ax.legend(fontsize=20, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'illumination_robustness_moderate_{adjust_mode}.png'), dpi=300)
    plt.close()

    # Plot 3: Heatmap of all results
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = []
    yticks = []
    for cat in categories:
        for diff in difficulties:
            row = []
            for r in all_results:
                if cat in r['metrics'] and diff in r['metrics'][cat]:
                    row.append(r['metrics'][cat][diff])
                else:
                    row.append(0)
            heatmap_data.append(row)
            yticks.append(f'{cat}-{difficulty_labels[diff]}')

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f'{v:.2f}' for v in x_values], rotation=45)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_title(f'Illumination Robustness Heatmap ({adjust_mode})', fontsize=14)

    # Add text annotations
    for i in range(len(yticks)):
        for j in range(len(x_values)):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.colorbar(im, ax=ax, label='AP (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'illumination_robustness_heatmap_{adjust_mode}.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Illumination Robustness Test for 3D Object Detection')
    parser.add_argument('--config', dest='config', required=True, help='settings in yaml format')
    parser.add_argument('--adjust_mode', type=str, default='brightness', choices=['gamma', 'brightness', 'both'],
                        help='Illumination adjustment mode. brightness: factor*pixel (factor<1=darker); gamma: pixel^gamma (gamma>1=darker)')
    parser.add_argument('--gamma_min', type=float, default=1.0, help='Minimum gamma value')
    parser.add_argument('--gamma_max', type=float, default=3.0, help='Maximum gamma value')
    parser.add_argument('--gamma_step', type=float, default=0.2, help='Gamma step size')
    parser.add_argument('--brightness_min', type=float, default=0.1, help='Minimum brightness factor')
    parser.add_argument('--brightness_max', type=float, default=1.0, help='Maximum brightness factor')
    parser.add_argument('--brightness_step', type=float, default=0.1, help='Brightness step size')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (override config)')
    args = parser.parse_args()

    assert os.path.exists(args.config), args.config
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    if args.output_dir is None:
        args.output_dir = os.path.join('./' + cfg["trainer"]['save_path'], model_name, 'illumination_test')
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, f'illumination_test_{args.adjust_mode}.log')
    logger = create_logger(log_file)

    # Determine test values: all go from max -> min (bright -> dark)
    if args.adjust_mode == 'gamma':
        # gamma > 1 = darker, so max->min means e.g. 3.0 -> 1.0 (dark -> bright)
        # We want bright -> dark, so go from gamma_min(1.0) -> gamma_max(3.0+)
        test_values = np.arange(args.gamma_min, args.gamma_max + 1e-6, args.gamma_step)
        test_values = [round(float(v), 3) for v in test_values]
    elif args.adjust_mode == 'brightness':
        # brightness < 1 = darker, so 1.0 -> 0.1 (bright -> dark)
        test_values = np.arange(args.brightness_max, args.brightness_min - 1e-6, -args.brightness_step)
        test_values = [round(float(v), 3) for v in test_values]
        # Explicitly append min value if not included due to float precision
        if args.brightness_min not in test_values:
            test_values.append(round(float(args.brightness_min), 3))
    else:  # both
        test_values = np.arange(args.brightness_max, args.brightness_min - 1e-6, -args.brightness_step)
        test_values = [round(float(v), 3) for v in test_values]
        if args.brightness_min not in test_values:
            test_values.append(round(float(args.brightness_min), 3))

    logger.info(f'==================== Illumination Robustness Test ====================')
    logger.info(f'Mode: {args.adjust_mode}')
    logger.info(f'Test values: {test_values}')
    logger.info(f'Output dir: {args.output_dir}')

    # Build model (once)
    model, _ = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    # Load checkpoint
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name, "checkpoint_best.pth")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    load_checkpoint(model=model, optimizer=None, filename=checkpoint_path,
                    map_location=device, logger=logger)
    model.to(device)
    model.eval()

    all_results = []

    for val in test_values:
        logger.info(f'\n{"="*60}')
        if args.adjust_mode == 'gamma':
            logger.info(f'Testing gamma = {val}')
            gamma = val
            brightness = 1.0
        elif args.adjust_mode == 'brightness':
            logger.info(f'Testing brightness = {val}')
            gamma = 1.0
            brightness = val
        else:
            logger.info(f'Testing gamma = {val}, brightness = {val}')
            gamma = val
            brightness = val

        # Build dataloader with current illumination setting
        test_loader = build_illumination_dataloader(
            cfg['dataset'], gamma=gamma, brightness_factor=brightness, adjust_mode=args.adjust_mode
        )

        # Create tester
        tester = Tester(
            cfg=cfg['tester'],
            model=model,
            dataloader=test_loader,
            logger=logger,
            train_cfg=cfg['trainer'],
            model_name=model_name
        )

        # Run inference and evaluation
        tester.inference_with_stereo()
        car_moderate = tester.evaluate()

        # Parse log for detailed metrics
        # Re-read log file to get the latest evaluation results
        metrics = parse_eval_log(log_file)

        result_entry = {
            'gamma': gamma,
            'brightness': brightness,
            'car_moderate': float(car_moderate) if car_moderate is not None else 0.0,
            'metrics': metrics
        }
        all_results.append(result_entry)

        # Save intermediate results
        with open(os.path.join(args.output_dir, f'results_{args.adjust_mode}.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Final summary
    logger.info(f'\n{"="*60}')
    logger.info('SUMMARY:')
    for r in all_results:
        if args.adjust_mode == 'gamma':
            logger.info(f"Gamma={r['gamma']:.2f}: Car_mod={r['car_moderate']:.2f}")
        else:
            logger.info(f"Brightness={r['brightness']:.2f}: Car_mod={r['car_moderate']:.2f}")

    # Plot results
    plot_results(all_results, args.output_dir, args.adjust_mode)

    logger.info(f'\nAll results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
