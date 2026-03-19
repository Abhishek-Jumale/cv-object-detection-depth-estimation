"""
Computer Vision - Task 2: Object Detection and Depth Estimation
RWU Hochschule Ravensburg-Weingarten
Prof. Dr. Stefan Elser

Outputs:
- Annotated images with red (YOLO) / green (GT) boxes, IoU, distances
- Precision & Recall per image
- Scatter plot: estimated vs ground truth distance
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "KITTI_Selection"  # Current directory with the dataset
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
CALIB_DIR = os.path.join(BASE_DIR, "calib")
OUTPUT_DIR = "output"

CAMERA_HEIGHT = 1.65      # meters above road
IOU_THRESHOLD = 0.5
YOLO_MODEL = "yolov8x.pt" # yolov8x for best results (as in the report)

# Minimum confidence threshold for YOLO detections (lower = more detections)
CONFIDENCE_THRESHOLD = 0.3  # Default is 0.25, lower for more cars


# ============================================================
# DATA LOADING
# ============================================================

def load_intrinsic_matrix(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('P2:') or line.startswith('P2 '):
            values = line.strip().split()[1:]
            P = np.array([float(v) for v in values]).reshape(3, 4)
            return P[:, :3]
    # Fallback: try to read any matrix-like content
    # This handles both "v1 v2 v3 v4 v5 v6 v7 v8 v9" and multi-line 3x3 matrices
    values = []
    for line in lines:
        parts = line.strip().split()
        if parts:  # Skip empty lines
            try:
                values.extend([float(v) for v in parts])
            except ValueError:
                continue
    
    if len(values) >= 9:
        K = np.array(values[:9]).reshape(3, 3)
        return K
    
    print(f"  WARNING: Could not parse calibration file: {calib_path}")
    return None


def load_labels(label_path):
    objects = []
    if not os.path.exists(label_path):
        return objects
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6 or parts[0] != 'Car':
            continue
        try:
            bbox = [float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])]
            distance_gt = float(parts[5])
            objects.append({'bbox': bbox, 'distance_gt': distance_gt})
        except (ValueError, IndexError):
            continue
    return objects


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_detection(image_path, model):
    results = model(image_path, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            if cls_id == 2 and conf >= CONFIDENCE_THRESHOLD:  # COCO class 2 = car, with confidence filter
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf
                })
    return detections


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    inter = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# MATCHING
# ============================================================

def match_detections(detections, gt_objects):
    if not detections or not gt_objects:
        return [], list(range(len(detections))), list(range(len(gt_objects))), [0.0]*len(detections)

    n_det, n_gt = len(detections), len(gt_objects)
    iou_matrix = np.zeros((n_det, n_gt))
    for i in range(n_det):
        for j in range(n_gt):
            iou_matrix[i, j] = calculate_iou(detections[i]['bbox'], gt_objects[j]['bbox'])

    matched, used_det, used_gt = [], set(), set()
    det_best_iou = [0.0] * n_det

    temp = iou_matrix.copy()
    while True:
        max_val = temp.max()
        if max_val < IOU_THRESHOLD:
            break
        i, j = np.unravel_index(temp.argmax(), temp.shape)
        if i not in used_det and j not in used_gt:
            matched.append({
                'det_idx': i, 'gt_idx': j, 'iou': max_val,
                'det_bbox': detections[i]['bbox'],
                'gt_bbox': gt_objects[j]['bbox'],
                'gt_distance': gt_objects[j]['distance_gt']
            })
            used_det.add(i)
            used_gt.add(j)
            det_best_iou[i] = max_val
        temp[i, j] = 0

    for i in range(n_det):
        if i not in used_det:
            det_best_iou[i] = iou_matrix[i].max() if n_gt > 0 else 0.0

    fp_idx = [i for i in range(n_det) if i not in used_det]
    fn_idx = [j for j in range(n_gt) if j not in used_gt]
    return matched, fp_idx, fn_idx, det_best_iou


def calc_precision_recall(matched, fp_idx, fn_idx):
    tp, fp, fn = len(matched), len(fp_idx), len(fn_idx)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec, tp, fp, fn


# ============================================================
# DISTANCE ESTIMATION
# ============================================================

def estimate_distance(bbox, K):
    """
    Midpoint = ((x1+x2)/2, y2)
    p = [x_new, y2, 1]
    d = K^-1 * p
    Z = camera_height / d_y
    X = Z * d_x,  Y = Z * d_y
    Distance = sqrt(X^2 + Y^2 + Z^2)
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = y2

    p = np.array([u, v, 1.0])
    d = np.linalg.inv(K) @ p

    if d[1] <= 0:
        return None

    Z = CAMERA_HEIGHT / d[1]
    X = Z * d[0]
    Y = Z * d[1]

    return np.sqrt(X**2 + Y**2 + Z**2)


# ============================================================
# VISUALIZATION (report style)
# ============================================================

def visualize_scene(image, detections, gt_objects, matched,
                    fp_idx, fn_idx, det_best_iou,
                    K, scene, precision, recall, output_dir):
    vis = image.copy()

    # Title bar: "Image - Precision: X.XX, Recall: X.XX"
    title = f"Image - Precision: {precision:.2f}, Recall: {recall:.2f}"
    cv2.rectangle(vis, (0, 0), (len(title)*9 + 10, 22), (255, 255, 255), -1)
    cv2.putText(vis, title, (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Green = GT boxes
    for gt in gt_objects:
        pt1 = (int(gt['bbox'][0]), int(gt['bbox'][1]))
        pt2 = (int(gt['bbox'][2]), int(gt['bbox'][3]))
        cv2.rectangle(vis, pt1, pt2, (0, 255, 0), 2)

    # Red = YOLO boxes
    for det in detections:
        pt1 = (int(det['bbox'][0]), int(det['bbox'][1]))
        pt2 = (int(det['bbox'][2]), int(det['bbox'][3]))
        cv2.rectangle(vis, pt1, pt2, (0, 0, 255), 2)

    # Build match lookup
    match_map = {m['det_idx']: m for m in matched}

    # Labels for each detection
    for i, det in enumerate(detections):
        x1 = int(det['bbox'][0])
        y1 = int(det['bbox'][1])
        iou_val = det_best_iou[i]
        d_calc = estimate_distance(det['bbox'], K)
        d_str = f"{d_calc:.2f}m" if d_calc else "N/A"

        if i in match_map:
            d_gt = match_map[i]['gt_distance']
            lines = [f"IoU: {iou_val:.2f}", f"YOLO: {d_str}", f"GT: {d_gt:.2f}m"]
        else:
            lines = [f"IoU: {iou_val:.2f}", f"YOLO: {d_str}"]

        # Draw text labels above box
        for j, txt in enumerate(reversed(lines)):
            ty = y1 - 5 - j * 14
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(vis, (x1, ty - th - 1), (x1 + tw + 3, ty + 2),
                          (255, 255, 255), -1)
            cv2.putText(vis, txt, (x1 + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{scene}.png")
    cv2.imwrite(path, vis)
    return path


def plot_scatter(pairs, output_dir):
    if not pairs:
        return
    d_calc = [p[0] for p in pairs]
    d_gt = [p[1] for p in pairs]

    plt.figure(figsize=(8, 8))
    plt.scatter(d_gt, d_calc, c='blue', s=30, alpha=0.7, label='tuple(YOLO,GT)', zorder=5)
    plt.plot([0, 100], [0, 100], color='orange', lw=1.5, label='Perfect Prediction', zorder=3)
    plt.xlabel('Distance provided in ground truth (m)', fontsize=12)
    plt.ylabel('Distance calculated using camera information (m)', fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "distance_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nScatter plot saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    for d, name in [(IMAGE_DIR, "images"), (LABEL_DIR, "labels"), (CALIB_DIR, "calib")]:
        if not os.path.exists(d):
            print(f"ERROR: '{name}' not found at: {os.path.abspath(d)}")
            print(f"Set BASE_DIR in this script to your KITTI_Selected folder path.")
            return

    print(f"Loading YOLO model: {YOLO_MODEL}")
    print("(First run downloads the model - wait for it to finish)\n")
    model = YOLO(YOLO_MODEL)

    images = sorted([f for f in os.listdir(IMAGE_DIR)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(images)} images.\n")

    all_results = []
    all_pairs = []

    for img_file in images:
        scene = os.path.splitext(img_file)[0]
        print(f"{'='*60}\nScene: {scene}")

        image = cv2.imread(os.path.join(IMAGE_DIR, img_file))
        if image is None:
            continue

        K = load_intrinsic_matrix(os.path.join(CALIB_DIR, scene + '.txt'))
        if K is None:
            print(f"  ERROR: No calibration for {scene}")
            continue

        gt = load_labels(os.path.join(LABEL_DIR, scene + '.txt'))
        dets = run_yolo_detection(os.path.join(IMAGE_DIR, img_file), model)

        print(f"  GT cars: {len(gt)}, YOLO cars: {len(dets)}")

        matched, fp_idx, fn_idx, det_iou = match_detections(dets, gt)
        prec, rec, tp, fp, fn = calc_precision_recall(matched, fp_idx, fn_idx)
        print(f"  TP={tp} FP={fp} FN={fn}  Precision={prec:.2f} Recall={rec:.2f}")

        for m in matched:
            d_c = estimate_distance(m['det_bbox'], K)
            d_g = m['gt_distance']
            if d_c is not None:
                all_pairs.append((d_c, d_g))
                print(f"  -> IoU={m['iou']:.2f}  YOLO={d_c:.2f}m  GT={d_g:.2f}m  err={abs(d_c-d_g):.2f}m")

        vis = visualize_scene(image, dets, gt, matched, fp_idx, fn_idx,
                              det_iou, K, scene, prec, rec, OUTPUT_DIR)
        print(f"  Saved: {vis}")

        all_results.append({'scene': scene, 'prec': prec, 'rec': rec,
                            'tp': tp, 'fp': fp, 'fn': fn,
                            'n_gt': len(gt), 'n_det': len(dets)})

    # Summary
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'Scene':<15} {'GT':>3} {'Det':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'Prec':>6} {'Rec':>6}")
    print("-"*70)
    s_tp = s_fp = s_fn = 0
    for r in all_results:
        print(f"{r['scene']:<15} {r['n_gt']:>3} {r['n_det']:>4} "
              f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['prec']:>6.2f} {r['rec']:>6.2f}")
        s_tp += r['tp']; s_fp += r['fp']; s_fn += r['fn']
    p = s_tp/(s_tp+s_fp) if (s_tp+s_fp)>0 else 0
    r = s_tp/(s_tp+s_fn) if (s_tp+s_fn)>0 else 0
    print("-"*70)
    print(f"{'TOTAL':<15} {'':>3} {'':>4} {s_tp:>3} {s_fp:>3} {s_fn:>3} {p:>6.2f} {r:>6.2f}")

    if all_pairs:
        errs = [abs(a-b) for a,b in all_pairs]
        print(f"\nDISTANCE STATS: {len(all_pairs)} cars | "
              f"mean={np.mean(errs):.2f}m median={np.median(errs):.2f}m max={np.max(errs):.2f}m")

    plot_scatter(all_pairs, OUTPUT_DIR)
    print(f"\nAll results in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()