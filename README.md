# Object Detection and Depth Estimation using KITTI Dataset

**Computer Vision — Task 2**
RWU Hochschule Ravensburg-Weingarten | Prof. Dr. Stefan Elser

## Overview

This project estimates the 3D depth of cars detected in 2D images using the KITTI dataset. It combines YOLOv8x object detection with pinhole camera geometry to calculate real-world distances from a single monocular camera.

## Pipeline

```
Input Image → YOLOv8x Detection → Filter Cars (COCO class) → IoU Matching with GT
    → Precision & Recall → Distance Estimation (Intrinsic Matrix) → Evaluation
```

## How It Works

### Object Detection
- YOLOv8x (pretrained on COCO) detects objects in each KITTI scene
- Only `car` class detections are kept

### Detection Evaluation
- **IoU** (Intersection over Union) measures overlap between predicted and ground truth bounding boxes
- Detections with IoU ≥ 0.5 are considered True Positives
- **Precision** and **Recall** are computed per scene

### Depth Estimation
Using the camera's intrinsic matrix K and known mounting height (1.65m):

1. Take the bottom-center pixel of the bounding box: `p = ((x1+x2)/2, y2, 1)`
2. Back-project to 3D ray: `d = K⁻¹ · p`
3. Intersect with ground plane: `Z = 1.65 / d_y`
4. Compute 3D point: `X = Z·d_x, Y = Z·d_y`
5. Distance = `√(X² + Y² + Z²)`

## Dataset

Uses a subset of the [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/) with 20 selected scenes.

```
KITTI_Selection/KITTI_Selected/
├── calib/      # Calibration files (intrinsic matrices)
├── images/     # Input images (.png)
├── labels/     # Ground truth labels (car bounding boxes + distances)
└── README      # Dataset description
```

> The dataset is not included in this repository. Download it from the course Moodle page.

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Abhishek-Jumale/cv-object-detection-depth-estimation.git
cd cv-object-detection-depth-estimation

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Open `main.py` and set `BASE_DIR` to point to your KITTI dataset:

```python
BASE_DIR = "KITTI_Selection/KITTI_Selected"  # adjust this path
```

### Run

```bash
python main.py
```

The first run downloads the YOLOv8x model (~130MB).

## Output

All results are saved in the `output/` folder:

| File | Description |
|------|-------------|
| `<scene_name>.png` | Annotated image with red (YOLO) and green (GT) bounding boxes, IoU values, and distance labels |
| `distance_comparison.png` | Scatter plot comparing estimated vs ground truth distances |

### Sample Output

Each annotated image shows:
- **Green boxes**: Ground truth bounding boxes
- **Red boxes**: YOLOv8 detections
- **Labels**: IoU score, YOLO estimated distance, GT distance
- **Title bar**: Precision and Recall for that scene

## Results

- Distance estimation works well for unoccluded cars on flat roads (error < 2m)
- Accuracy decreases with occlusion, crowded scenes, slopes, and distant objects
- Scatter plot shows most points clustered near the diagonal (perfect prediction line)

### Known Limitations

| Issue | Effect |
|-------|--------|
| Occluded cars | Bounding box bottom edge doesn't match ground contact point |
| Crowded scenes | Overlapping detections shift the reference point |
| Non-flat terrain | Ground plane assumption breaks down |
| Distant cars | Small pixel errors cause large distance errors |

## Project Structure

```
.
├── main.py              # Main script (detection, evaluation, visualization)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
└── output/              # Generated results (not tracked)
```

## References

1. Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.* CVPR.
2. Ultralytics. *YOLOv8* [software]. Available at: https://docs.ultralytics.com/
3. Elser, S. *3D to 2D Projections, 2D to 3D Projections and 3D Rotations.* Chapter 7, Computer Vision Lecture.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file.
