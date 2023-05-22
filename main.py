import os
import numpy as np
import pickle
import pathlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import WDDDataset
from model import WDDModel
from helper import timeit

PATH_PICKLE = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
PATH_IMAGES = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

params = {
    "batch_size": 64,
    "num_workers": 8,
    "num_epochs": 32,
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

    train_dataset = WDDDataset(gt_train_items)
    assert len(train_dataset) == len(train_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], num_workers=params["num_workers"]) 

    model = WDDModel(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(params["num_epochs"]):
        model.train()
        for batch_idx, (images, vector, duration, label) in enumerate(train_dataloader):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, label)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Batch: {batch_idx}")
            print(f"Loss: {loss:.4f}")

        model.eval()
        with torch.no_grad():
            correct, num_examples = 0, 0
            for i, (images, vector, duration, label) in enumerate(train_dataloader):
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                num_examples += logits.size(0) # batch size
                correct += (predicted == label).sum()
            acc = correct / num_examples * 100
            print(f"Epoch {epoch}")
            print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
