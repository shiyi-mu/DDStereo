import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.kitti_utils import Calibration, Object3d


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 128, 0),
    'Cyclist': (0, 128, 255),
    'OoD': (0, 0, 255),
}


def draw_3d_box(img, corners_2d, color, thickness=2):
    """
    corners_2d: (8, 2) projected 2D corners
    order: [front_bottom_left, front_bottom_right, front_top_right, front_top_left,
            back_bottom_left, back_bottom_right, back_top_right, back_top_left]
    Actually KITTI standard order after rotation:
    """
    corners_2d = corners_2d.astype(int)

    # Define edges: each edge connects two corner indices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # front face
        [4, 5], [5, 6], [6, 7], [7, 4],  # back face
        [0, 4], [1, 5], [2, 6], [3, 7],  # connecting edges
    ]

    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # Draw front face with thicker lines
    front_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for edge in front_edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(img, pt1, pt2, color, thickness + 1, lineType=cv2.LINE_AA)


def draw_2d_box(img, box2d, color, thickness=2):
    """box2d: [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = [int(v) for v in box2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_label(img, box2d, text, color, font_scale=0.6):
    """Draw label text above the 2D box"""
    x1, y1, x2, y2 = [int(v) for v in box2d]
    # Text background
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def parse_result_line(line):
    """Parse one line of KITTI format detection result"""
    parts = line.strip().split()
    cls_type = parts[0]
    truncation = float(parts[1])
    occlusion = float(parts[2])
    alpha = float(parts[3])
    x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
    h, w, l = [float(v) for v in parts[8:11]]
    x, y, z = [float(v) for v in parts[11:14]]
    ry = float(parts[14])
    score = float(parts[15])
    return {
        'cls_type': cls_type,
        'truncation': truncation,
        'occlusion': occlusion,
        'alpha': alpha,
        'box2d': [x1, y1, x2, y2],
        'h': h, 'w': w, 'l': l,
        'pos': np.array([x, y, z]),
        'ry': ry,
        'score': score,
    }


def create_corners3d(h, w, l, pos, ry):
    """Generate 3D box corners in camera coordinates"""
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + pos
    return corners3d


def visualize_one_image(img_id, image_dir, calib_dir, result_path, output_path, show_score=True, score_threshold=0.0):
    """Visualize detection results for a single image"""
    # Load image
    img_file = os.path.join(image_dir, f'{img_id}.png')
    if not os.path.exists(img_file):
        print(f"Image not found: {img_file}")
        return False

    img = np.array(Image.open(img_file).convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Load calib
    calib_file = os.path.join(calib_dir, f'{img_id}.txt')
    if not os.path.exists(calib_file):
        print(f"Calib not found: {calib_file}")
        return False
    calib = Calibration(calib_file)

    # Load results
    if not os.path.exists(result_path):
        print(f"Result not found: {result_path}")
        return False

    with open(result_path, 'r') as f:
        lines = f.readlines()

    # Draw detections
    for line in lines:
        det = parse_result_line(line)

        if det['score'] < score_threshold:
            continue

        cls_type = det['cls_type']
        color = CLASS_COLORS.get(cls_type, (128, 128, 128))

        # Generate 3D corners and project to 2D
        corners3d = create_corners3d(det['h'], det['w'], det['l'], det['pos'], det['ry'])
        corners2d, _ = calib.rect_to_img(corners3d)

        # Draw 3D box
        draw_3d_box(img, corners2d, color, thickness=2)

        # Draw 2D box
        draw_2d_box(img, det['box2d'], color, thickness=2)

        # Draw label
        label_text = cls_type
        if show_score:
            label_text += f' {det["score"]:.2f}'
        draw_label(img, det['box2d'], label_text, color)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D detection results on KITTI images')
    parser.add_argument('--result_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/outputs/data',
                        help='Directory containing detection result .txt files')
    parser.add_argument('--image_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/image_2',
                        help='Directory containing images')
    parser.add_argument('--calib_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/calib',
                        help='Directory containing calibration files')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/visualizations',
                        help='Output directory for visualization images')
    parser.add_argument('--img_ids', type=str, default=None,
                        help='Comma-separated list of image IDs to visualize (e.g., 000000,000001). If not set, visualize all.')
    parser.add_argument('--max_num', type=int, default=None,
                        help='Maximum number of images to visualize')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Filter detections with score below this threshold')
    parser.add_argument('--no_score', action='store_true',
                        help='Do not show confidence scores on labels')
    args = parser.parse_args()

    # Determine which images to visualize
    if args.img_ids:
        img_ids = [sid.strip() for sid in args.img_ids.split(',')]
    else:
        result_files = sorted([f for f in os.listdir(args.result_dir) if f.endswith('.txt')])
        img_ids = [f[:-4] for f in result_files]

    if args.max_num:
        img_ids = img_ids[:args.max_num]

    print(f"Visualizing {len(img_ids)} images...")

    success_count = 0
    for img_id in img_ids:
        result_path = os.path.join(args.result_dir, f'{img_id}.txt')
        output_path = os.path.join(args.output_dir, f'{img_id}.png')
        ret = visualize_one_image(
            img_id=img_id,
            image_dir=args.image_dir,
            calib_dir=args.calib_dir,
            result_path=result_path,
            output_path=output_path,
            show_score=not args.no_score,
            score_threshold=args.score_threshold
        )
        if ret:
            success_count += 1

    print(f"Done. Successfully visualized {success_count}/{len(img_ids)} images.")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
