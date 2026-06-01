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

CLASS_COLORS = {
    'Car': (0, 200, 0),
    'Pedestrian': (255, 128, 0),
    'Cyclist': (0, 128, 255),
    'OoD': (0, 0, 255),
    'Ood': (0, 0, 255),
}


def draw_2d_box(img, box2d, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_label(img, box2d, text, color, font_scale=0.5):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    # place label above the box if possible, otherwise below
    label_y = y1 - 4 if y1 > text_h + 10 else y2 + text_h + 4
    cv2.rectangle(img, (x1, label_y - text_h - 2), (x1 + text_w + 4, label_y + 2), color, -1)
    cv2.putText(img, text, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def visualize_one_image(img_id, image_dir, result_path, output_path, score_threshold=0.0, show_score=True):
    img_file = os.path.join(image_dir, f'{img_id}.png')
    if not os.path.exists(img_file):
        return False

    img = np.array(Image.open(img_file).convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if not os.path.exists(result_path):
        # save original image even if no detections
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)
        return True

    with open(result_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 16:
            continue
        cls_type = parts[0]
        score = float(parts[-1])
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
        color = CLASS_COLORS.get(cls_type, (128, 128, 128))

        draw_2d_box(img, [x1, y1, x2, y2], color, thickness=2)

        label_text = cls_type
        if show_score:
            label_text += f' {score:.2f}'
        draw_label(img, [x1, y1, x2, y2], label_text, color)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize 2D detection results')
    parser.add_argument('--result_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/outputs/data',
                        help='Directory containing detection result .txt files')
    parser.add_argument('--image_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/image_2',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/visualizations_2d',
                        help='Output directory')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Filter detections with score below this threshold')
    parser.add_argument('--no_score', action='store_true',
                        help='Do not show confidence scores')
    parser.add_argument('--max_num', type=int, default=None,
                        help='Max number of images to visualize')
    args = parser.parse_args()

    result_files = sorted([f for f in os.listdir(args.result_dir) if f.endswith('.txt')])
    img_ids = [f[:-4] for f in result_files]

    if args.max_num:
        img_ids = img_ids[:args.max_num]

    print(f"Visualizing {len(img_ids)} images...")
    success = 0
    for img_id in tqdm(img_ids, desc="Processing"):
        result_path = os.path.join(args.result_dir, f'{img_id}.txt')
        output_path = os.path.join(args.output_dir, f'{img_id}.png')
        ret = visualize_one_image(
            img_id=img_id,
            image_dir=args.image_dir,
            result_path=result_path,
            output_path=output_path,
            score_threshold=args.score_threshold,
            show_score=not args.no_score
        )
        if ret:
            success += 1

    print(f"Done. {success}/{len(img_ids)} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
