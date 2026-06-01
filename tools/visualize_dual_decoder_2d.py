import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

MONO_COLORS = {
    'Car': (0, 200, 0),
    'Pedestrian': (255, 128, 0),
    'Cyclist': (0, 128, 255),
    'OoD': (0, 0, 255),
    'Ood': (0, 0, 255),
}

STEREO_COLOR = (0, 140, 255)  # orange in BGR


def draw_box(img, box2d, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_label(img, box2d, text, color, font_scale=0.5, offset_y=-4):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bg_y1 = y1 + offset_y - text_h - 2 if y1 > text_h + 10 else y2 + text_h + 4
    bg_y2 = bg_y1 + text_h + 4
    cv2.rectangle(img, (x1, bg_y1), (x1 + text_w + 4, bg_y2), color, -1)
    cv2.putText(img, text, (x1 + 2, bg_y2 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def parse_pred_line(line):
    parts = line.strip().split()
    cls_type = parts[0]
    score = float(parts[-1])
    x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
    return {'cls_type': cls_type, 'score': score, 'box2d': [x1, y1, x2, y2]}


def parse_stereo_line(line):
    parts = line.strip().split()
    score = float(parts[-1])
    x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
    return {'score': score, 'box2d': [x1, y1, x2, y2]}


def visualize_one(img_id, image_dir, mono_dir, stereo_dir, output_path,
                  mono_threshold=0.0, stereo_threshold=0.0):
    img_file = os.path.join(image_dir, f'{img_id}.png')
    if not os.path.exists(img_file):
        return False

    img = np.array(Image.open(img_file).convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw stereo branch (foreground detector)
    stereo_path = os.path.join(stereo_dir, f'{img_id}.txt')
    if os.path.exists(stereo_path):
        with open(stereo_path, 'r') as f:
            for line in f:
                det = parse_stereo_line(line)
                if det['score'] < stereo_threshold:
                    continue
                draw_box(img, det['box2d'], STEREO_COLOR, thickness=2)
                draw_label(img, det['box2d'], f"S {det['score']:.2f}", STEREO_COLOR, font_scale=0.45)

    # Draw mono branch (final classifier)
    mono_path = os.path.join(mono_dir, f'{img_id}.txt')
    if os.path.exists(mono_path):
        with open(mono_path, 'r') as f:
            for line in f:
                det = parse_pred_line(line)
                if det['score'] < mono_threshold:
                    continue
                color = MONO_COLORS.get(det['cls_type'], (128, 128, 128))
                draw_box(img, det['box2d'], color, thickness=2)
                draw_label(img, det['box2d'], f"M {det['cls_type']} {det['score']:.2f}", color, font_scale=0.45)

    # Legend
    legend_h = 65
    legend_w = 220
    legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)
    cv2.putText(legend, "Decoder Comparison", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(legend, (10, 30), (40, 30), STEREO_COLOR, 2)
    cv2.putText(legend, "Stereo (FG)", (45, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.line(legend, (10, 50), (40, 50), MONO_COLORS['Car'], 2)
    cv2.putText(legend, "Mono (Car)", (45, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.line(legend, (120, 50), (150, 50), MONO_COLORS['OoD'], 2)
    cv2.putText(legend, "Mono (OoD)", (155, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    h, w = img.shape[:2]
    lx = w - legend_w - 10
    ly = 10
    img[ly:ly+legend_h, lx:lx+legend_w] = legend

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize both stereo and mono decoder 2D boxes')
    parser.add_argument('--mono_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/outputs/data',
                        help='Mono branch detection results')
    parser.add_argument('--stereo_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/outputs/data_stereo',
                        help='Stereo branch detection results')
    parser.add_argument('--image_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/image_2',
                        help='Image directory')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/visualizations_dual_decoder',
                        help='Output directory')
    parser.add_argument('--mono_threshold', type=float, default=0.0)
    parser.add_argument('--stereo_threshold', type=float, default=0.0)
    parser.add_argument('--max_num', type=int, default=None)
    args = parser.parse_args()

    mono_files = sorted([f for f in os.listdir(args.mono_dir) if f.endswith('.txt')])
    img_ids = [f[:-4] for f in mono_files]
    if args.max_num:
        img_ids = img_ids[:args.max_num]

    print(f"Visualizing {len(img_ids)} images...")
    success = 0
    for img_id in tqdm(img_ids, desc="Processing"):
        output_path = os.path.join(args.output_dir, f'{img_id}.png')
        ret = visualize_one(
            img_id=img_id,
            image_dir=args.image_dir,
            mono_dir=args.mono_dir,
            stereo_dir=args.stereo_dir,
            output_path=output_path,
            mono_threshold=args.mono_threshold,
            stereo_threshold=args.stereo_threshold
        )
        if ret:
            success += 1
    print(f"Done. {success}/{len(img_ids)} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
