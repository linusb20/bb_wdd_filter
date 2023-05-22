import os
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import WDDDataset

PATH_PICKLE = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
PATH_IMAGES = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

params = {
    "batch_size": 32,
    "num_workers": 8,
    "num_epochs": 512,
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
    assert len(dataset) == len(train_indices)

    # r = np.random.randint(len(train_indices))
    # images, vector, duration, label = dataset[r]
    # print(f"Training example {r}")
    # print(images.shape, vector, duration, label)
    # plt.imshow(images[0,1], cmap="gray")
    # plt.show()

    dataloader = DataLoader(dataset, batch_size=params["batch_size"], num_workers=params["num_workers"]) 
    for batch_idx, (images, vector, duration, label) in enumerate(dataloader):
        print(f"Batch Index: {batch_idx}")
        print("images:")
        print(type(images))
        print(images.size())
        print()

        print("vector:")
        print(type(vector))
        print(vector.size())
        print(vector)
        print()

        print("duration:")
        print(type(duration))
        print(duration.size())
        print(duration)
        print()

        print("label")
        print(type(label))
        print(label.size())
        print(label)
        print()

        plt.imshow(images[0,0,0], cmap="gray")
        plt.show()

        break

if __name__ == "__main__":
    main()
