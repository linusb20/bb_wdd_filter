import os
import numpy as np
import pickle
import pathlib
from torch.utils.data import DataLoader

from dataset import WDDDataset

PATH_PICKLE = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
PATH_IMAGES = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

params = {
    batch_size = 32
    num_workers = 8
    num_epochs = 512
}

def load_gt_items(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
        items = [(key,) + v for key, v in r.items()]
    return items

def main():
    gt_items = load_gt_items(PATH_PICKLE) 
    all_indices = np.arange(len(gt_items))
    mask = all_indices % 10 == 0
    test_indices = all_indices[mask]
    train_indices = all_indices[~mask]

    print(f"Found {len(test_indices)} test examples")
    print(f"Found {len(train_indices)} training examples")

    gt_train_items = [gt_items[i] for i in train_indices]

    def remap(p):
        head = pathlib.Path(PATH_IMAGES)
        tail = p.relative_to("/mnt/curta/storage/beesbook/wdd/")
        return head.joinpath(tail)

    gt_train_items = [tuple(item) + (remap(path),) for *item, path in gt_train_items]

    dataset = WDDDataset(gt_train_items)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers) 


if __name__ == "__main__":
    main()
