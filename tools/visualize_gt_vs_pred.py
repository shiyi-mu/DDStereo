import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.kitti_utils import Calibration

# GT colors (lighter, thinner)
GT_COLORS = {
    'Car': (180, 255, 180),
    'Pedestrian': (255, 200, 150),
    'Cyclist': (150, 200, 255),
    'OoD': (255, 150, 150),
    'Ood': (255, 150, 150),
}

# Pred colors (brighter, thicker)
PRED_COLORS = {
    'Car': (0, 180, 0),
    'Pedestrian': (255, 100, 0),
    'Cyclist': (0, 100, 255),
    'OoD': (0, 0, 200),
    'Ood': (0, 0, 200),
}


def draw_3d_box(img, corners_2d, color, thickness=2, dashed=False):
    corners_2d = corners_2d.astype(int)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        if dashed:
            draw_dashed_line(img, pt1, pt2, color, thickness)
        else:
            cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # Front face thicker
    if not dashed:
        front_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        for edge in front_edges:
            pt1 = tuple(corners_2d[edge[0]])
            pt2 = tuple(corners_2d[edge[1]])
            cv2.line(img, pt1, pt2, color, thickness + 1, lineType=cv2.LINE_AA)


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=12):
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    n_dashes = max(int(dist / dash_len), 1)
    for i in range(0, n_dashes * 2, 2):
        s = i / (n_dashes * 2)
        e = min((i + 1) / (n_dashes * 2), 1.0)
        sx = int(x1 + (x2 - x1) * s)
        sy = int(y1 + (y2 - y1) * s)
        ex = int(x1 + (x2 - x1) * e)
        ey = int(y1 + (y2 - y1) * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)


def draw_2d_box(img, box2d, color, thickness=2, dashed=False):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    if dashed:
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i in range(4):
            draw_dashed_line(img, pts[i], pts[(i+1)%4], color, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_label(img, box2d, text, color, font_scale=0.5, is_gt=False):
    x1, y1, x2, y2 = [int(v) for v in box2d]
    prefix = "GT:" if is_gt else "P:"
    full_text = prefix + text
    (text_w, text_h), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bg_y1 = y1 - text_h - 6 if y1 > text_h + 10 else y2 + text_h + 6
    bg_y2 = bg_y1 + text_h + 6
    cv2.rectangle(img, (x1, bg_y1), (x1 + text_w + 4, bg_y2), color, -1)
    cv2.putText(img, full_text, (x1 + 2, bg_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def parse_kitti_line(line):
    parts = line.strip().split()
    cls_type = parts[0]
    x1, y1, x2, y2 = [float(v) for v in parts[4:8]]
    h, w, l = [float(v) for v in parts[8:11]]
    x, y, z = [float(v) for v in parts[11:14]]
    ry = float(parts[14])
    score = float(parts[15]) if len(parts) > 15 else 1.0
    return {
        'cls_type': cls_type,
        'box2d': [x1, y1, x2, y2],
        'h': h, 'w': w, 'l': l,
        'pos': np.array([x, y, z]),
        'ry': ry,
        'score': score,
    }


def create_corners3d(h, w, l, pos, ry):
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])
    corners3d = np.dot(R, corners3d).T + pos
    return corners3d


def visualize_comparison(img_id, image_dir, calib_dir, gt_dir, pred_dir, output_dir,
                         score_threshold=0.0, show_gt_score=False, show_pred_score=True):
    img_file = os.path.join(image_dir, f'{img_id}.png')
    if not os.path.exists(img_file):
        return False
    img = np.array(Image.open(img_file).convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    calib_file = os.path.join(calib_dir, f'{img_id}.txt')
    if not os.path.exists(calib_file):
        return False
    calib = Calibration(calib_file)

    # Draw GT
    gt_path = os.path.join(gt_dir, f'{img_id}.txt')
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                det = parse_kitti_line(line)
                cls_type = det['cls_type']
                color = GT_COLORS.get(cls_type, (180, 180, 180))
                corners3d = create_corners3d(det['h'], det['w'], det['l'], det['pos'], det['ry'])
                corners2d, _ = calib.rect_to_img(corners3d)
                draw_3d_box(img, corners2d, color, thickness=1, dashed=True)
                draw_2d_box(img, det['box2d'], color, thickness=1, dashed=True)
                label_text = cls_type
                if show_gt_score:
                    label_text += f' {det["score"]:.2f}'
                draw_label(img, det['box2d'], label_text, color, is_gt=True)

    # Draw Pred
    pred_path = os.path.join(pred_dir, f'{img_id}.txt')
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            for line in f:
                det = parse_kitti_line(line)
                if det['score'] < score_threshold:
                    continue
                cls_type = det['cls_type']
                color = PRED_COLORS.get(cls_type, (128, 128, 128))
                corners3d = create_corners3d(det['h'], det['w'], det['l'], det['pos'], det['ry'])
                corners2d, _ = calib.rect_to_img(corners3d)
                draw_3d_box(img, corners2d, color, thickness=2, dashed=False)
                draw_2d_box(img, det['box2d'], color, thickness=2, dashed=False)
                label_text = cls_type
                if show_pred_score:
                    label_text += f' {det["score"]:.2f}'
                draw_label(img, det['box2d'], label_text, color, is_gt=False)

    # Legend
    legend_h = 90
    legend_w = 280
    legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)
    cv2.putText(legend, "Legend", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y = 40
    for name, (gt_c, pred_c) in [("Car", (GT_COLORS['Car'], PRED_COLORS['Car'])),
                                   ("OoD", (GT_COLORS['OoD'], PRED_COLORS['OoD']))]:
        cv2.line(legend, (10, y), (40, y), gt_c, 1)
        draw_dashed_line(legend, (10, y), (40, y), gt_c, 1)
        cv2.line(legend, (50, y), (80, y), pred_c, 2)
        cv2.putText(legend, f"GT {name}", (85, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 22
        cv2.line(legend, (50, y), (80, y), pred_c, 2)
        cv2.putText(legend, f"Pred {name}", (85, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 22

    h, w = img.shape[:2]
    lx = w - legend_w - 10
    ly = 10
    img[ly:ly+legend_h, lx:lx+legend_w] = legend

    output_path = os.path.join(output_dir, f'{img_id}.png')
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='GT vs Pred 3D detection comparison')
    parser.add_argument('--gt_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/label_2',
                        help='Ground truth label directory')
    parser.add_argument('--pred_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/outputs/data',
                        help='Prediction result directory')
    parser.add_argument('--image_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/image_2',
                        help='Image directory')
    parser.add_argument('--calib_dir', type=str,
                        default='/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/08-IoTJ/LaF_kitti-resize/calib',
                        help='Calibration directory')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/101-infer—on-laf/lightstereoOoDdetr/visualizations_gt_pred',
                        help='Output directory')
    parser.add_argument('--img_ids', type=str, default=None,
                        help='Comma-separated image IDs. If not set, use all pred results.')
    parser.add_argument('--max_num', type=int, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.0)
    args = parser.parse_args()

    if args.img_ids:
        img_ids = [sid.strip() for sid in args.img_ids.split(',')]
    else:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.txt')])
        img_ids = [f[:-4] for f in pred_files]

    if args.max_num:
        img_ids = img_ids[:args.max_num]

    print(f"Visualizing {len(img_ids)} images...")
    success = 0
    for img_id in img_ids:
        ret = visualize_comparison(
            img_id=img_id,
            image_dir=args.image_dir,
            calib_dir=args.calib_dir,
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            output_dir=args.output_dir,
            score_threshold=args.score_threshold
        )
        if ret:
            success += 1
    print(f"Done. {success}/{len(img_ids)} images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
