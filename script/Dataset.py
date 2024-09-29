"""
Script for Dataset Utilities
"""
from .ClassAverages import ClassAverages
from utilities.Calib import get_P
from torch.utils import data
from torchvision import transforms
import torch
import cv2
import numpy as np
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of bins

    return angle_bins


class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):
        # Configuring Paths related to stereo images, labels, and their calibration files.
        self.left_image_path = os.path.join(path, 'image_2')
        self.right_image_path = os.path.join(path, 'image_3')
        self.label_path = os.path.join(path, 'label_2')
        self.local_calib_path = os.path.join(path, 'calib')

        # Get index of image_2 files
        self.ids = [x.split('.')[0]
                    for x in sorted(os.listdir(self.local_calib_path))]
        self.num_images = len(self.ids)

        # Initializing other components.
        self.bins = bins
        self.angle_bins = generate_bins(self.bins)
        self.interval = 2 * np.pi / self.bins
        self.overlap = overlap

        # Ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(bins):
            self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi),
                                    (i * self.interval + self.interval + overlap) % (2 * np.pi)))

        # Calculate class averages for statistics summary.
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian',
                      'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        # List of object [id (000001), line_num]
        self.object_list = self.get_objects(self.ids)

        # Caching Mechanism to store processed labels and efficiently perform operations without duplications.
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id
            self.labels[id][str(line_num)] = label

        # Stores the current image being processed
        self.curr_id = ""
        self.curr_img_2 = None
        self.curr_img_disparity = None
        self.proj_matrix = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            left_img = cv2.imread(os.path.join(
                self.left_image_path, f'{id}.png'))
            right_img = cv2.imread(os.path.join(
                self.right_image_path, f'{id}.png'))

            if left_img is None or right_img is None:
                raise ValueError(f"Images not found for id: {id}")

            self.curr_img_2 = left_img.copy()
            self.curr_img_disparity = self.disparity(left_img, right_img)

            # Load the calibration matrix for the current image
            calib_file_path = os.path.join(self.local_calib_path, f'{id}.txt')
            self.proj_matrix = get_P(calib_file_path)

        label = self.labels[id][str(line_num)]

        obj = DetectedObject(
            self.curr_img_2, self.curr_img_disparity, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.combined, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):
        """
        Takes index/filename in the format (000002) and returns the labels and dimensions of an image. 
        """
        objects = []
        for id in ids:
            with open(os.path.join(self.label_path, f'{id}.txt')) as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(
                        line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)
                    objects.append((id, line_num))

        self.averages.dump_to_file()
        return objects

    def get_label(self, id, line_num):
        lines = open(os.path.join(self.label_path,
                     f'{id}.txt')).read().splitlines()
        label = self.format_label(lines[line_num])
        return label

    def get_bin(self, angle):
        bin_idxs = []

        def is_between(min_angle, max_angle, angle):
            max_angle = (max_angle - min_angle) if (max_angle -
                                                    min_angle) > 0 else (max_angle - min_angle) + 2 * np.pi
            angle = (angle - min_angle) if (angle -
                                            min_angle) > 0 else (angle - min_angle) + 2 * np.pi
            return angle < max_angle

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        # Alpha is orientation will be regressing
        # Alpha = [-pi, pi]
        Alpha = line[3]
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        # Dimension: height, width, length
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)
        # Modify the average
        Dimension -= self.averages.get_item(Class)

        # Location: x, y, z
        Location = [line[11], line[12], line[13]]
        # Bring the KITTI center up to the middle of the object
        Location[1] -= Dimension[0] / 2

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # Angle on range [0, 2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx, :] = np.array(
                [np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
            'Class': Class,
            'Box_2D': Box_2D,
            'Dimensions': Dimension,
            'Alpha': Alpha,
            'Orientation': Orientation,
            'Confidence': Confidence
        }

        return label

    def disparity(self, img_2, img_3):
        imgL = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)

        min_disp = 0
        num_disp = 16 * 8
        block_size = 9

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=6,        # Lower uniqueness ratio for better matches
            speckleWindowSize=30,     # Small speckle window size for finer details
            speckleRange=4,           # Narrow range to remove noise
            preFilterCap=9,          # Lower pre-filter cap for preserving details
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(imgL, imgR).astype(np.float32)

        return disparity


class DetectedObject:
    """
    Processing image and disparity map for NN input
    """

    def __init__(self, img, disparity_map, detection_class, box_2d, proj_matrix, label=None):
        # Check if proj_matrix is path
        if isinstance(proj_matrix, str):
            proj_matrix = get_P(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img, self.disparity = self.format_img_and_disparity(
            img, disparity_map, box_2d)
        self.combined = self.combine_inputs(self.img, self.disparity)

        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """
        Calculate global angle of object, see paper
        """
        width = img.shape[1]
        # Angle of View: fovx (rad) => 3.14
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def format_img_and_disparity(self, img, disparity_map, box_2d):
        # Transforms
        normalize_img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        normalize_disp = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )

        process_img = transforms.Compose([
            transforms.ToTensor(),
            normalize_img
        ])

        process_disp = transforms.Compose([
            transforms.ToTensor(),
            normalize_disp
        ])

        # Crop image
        pt1, pt2 = box_2d[0], box_2d[1]
        crop_img = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop_img = cv2.resize(crop_img, (224, 224),
                              interpolation=cv2.INTER_CUBIC)

        # Crop disparity map
        crop_disp = disparity_map[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop_disp = cv2.resize(crop_disp, (224, 224),
                               interpolation=cv2.INTER_CUBIC)
        crop_disp = cv2.normalize(crop_disp, None, 0, 1, cv2.NORM_MINMAX)

        # Apply transforms for batch processing
        batch_img = process_img(crop_img)
        batch_disp = process_disp(crop_disp)

        return batch_img, batch_disp

    def combine_inputs(self, img_tensor, disparity_tensor):
        # Concatenate the RGB image and the disparity map into a single 4-channel tensor
        # Ensure disparity_tensor is in the correct shape for concatenation
        # disparity_tensor should be (1, 224, 224) after ToTensor
        # We can add a new dimension to match the expected input shape
        disparity_tensor = disparity_tensor.unsqueeze(0)

        # Concatenate along the channel dimension
        # Result shape: (4, 224, 224)
        combined = torch.cat((img_tensor, disparity_tensor), dim=0)

        return combined


if __name__ == '__main__':
    train_path = ROOT / 'dataset/KITTI/training'
    dataset = Dataset(train_path)

    # Iterate through a few samples
    for i in range(min(5, len(dataset))):
        img, label = dataset[i]
        print(
            f"Image Shape: {img.shape}, Class: {label['Class']}, Box: {label['Box_2D']}")
