import os
import datetime
import torch

# PATH_PICKLE = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
# PATH_IMAGES = os.path.join(os.path.dirname(__file__), "wdd_ground_truth", "wdd_ground_truth")

PATH_PICKLE = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth", "ground_truth_wdd_angles.pickle")
PATH_IMAGES = os.path.join(os.sep, "srv", "data", "joeh97", "data", "wdd_ground_truth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_EPOCHS = 32

STATS_PATH = os.path.join(os.getcwd(), "stats_" + datetime.datetime.now().strftime("%Y%m%dT%H%M"))
SAVE_PATH_ACCURACY = os.path.join(STATS_PATH, "accuracy.pdf")
SAVE_PATH_LOSS = os.path.join(STATS_PATH, "loss.pdf")
SAVE_PATH_CONFUSION = os.path.join(STATS_PATH, "confusion.pdf")
SAVE_PATH_JSON = os.path.join(STATS_PATH, "stats.json")
SAVE_PATH_MODEL_SUMMARY = os.path.join(STATS_PATH, "model.txt")
