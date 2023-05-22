import os
import numpy as np
import pandas
import PIL.Image
import zipfile
import json
import skimage.transform as transform
from torch.utils.data import Dataset

class WDDDataset(Dataset):
    def __init__(self, gt_items):
        """
        Args:
            gt_items: list of 4-tuples of `waggle_id`, `label`, `gt_angle`, `path`
        """
        self.gt_df = pandas.DataFrame(gt_items, columns=["waggle_id", "label", "gt_angle", "path"])
        self.meta_data_paths = self.gt_df.path.values
        labels = self.gt_df.label.copy()
        labels[labels == "trembling"] = "other"
        self.all_labels = ["other", "waggle", "ventilating", "activating"]
        label_mapper = {s: i for i, s in enumerate(self.all_labels)}
        self.Y = np.array([label_mapper[l] for l in labels])

    def __len__(self):
        return len(self.meta_data_paths)

    def __getitem__(self, i):
        images, vector, duration = WDDDataset.load_waggle_metadata(self.meta_data_paths[i])
        label = self.Y[i]
        crop_size = 80  # smallest sequence length is 95
        crop_start = (len(images)//2) - (crop_size//2)
        crop_end = (len(images)//2) + (crop_size//2)
        images = images[crop_start:crop_end, :, :]
        images = np.expand_dims(images, axis=0)

        return images, vector, duration, label

    @staticmethod
    def load_image(f):
        img = PIL.Image.open(f)
        img = np.asarray(img, dtype=np.float32)
        img = img / 255 * 2 - 1 # normalize to [-1, 1]
        img = transform.resize(img, (60, 60))
        return img

    @staticmethod
    def load_waggle_metadata(waggle_path, include_images=True):
        if include_images:
            images = []
            waggle_dir = waggle_path.parent
            zip_file_path = os.path.join(waggle_dir, "images.zip")
            assert os.path.exists(zip_file_path) 
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                image_fns = zf.namelist()
                for fn in image_fns:
                    with zf.open(fn, "r") as f:
                        images.append(WDDDataset.load_image(f))
        images = np.asarray(images)
        with open(waggle_path, "r") as f:
            metadata = json.load(f)
        waggle_angle = metadata["waggle_angle"]
        waggle_duration = metadata["waggle_duration"]

        waggle_vector = np.array([np.cos(waggle_angle), np.sin(waggle_angle)], dtype=np.float32)
        return images, waggle_vector, waggle_duration
