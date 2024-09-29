from script.Dataset import Dataset

from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

train_path = ROOT / 'dataset/KITTI/training'
dataset = Dataset(train_path)

# Iterate through a few samples
for i in range(min(5, len(dataset))):
    img, label = dataset[i]
    print(
        f"Image Shape: {img.shape}, Class: {label['Class']}, Box: {label['Box_2D']}")
