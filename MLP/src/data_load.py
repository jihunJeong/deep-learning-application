import torch
import pickle
import gzip

import numpy as np
import requests

from pathlib import Path
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("../data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)
FILENAME = "mnist.pkl.gz"

# download path 정의
download_root = '../data'

train_dataset = MNIST(download_root, train=True, download=True)
valid_dataset = MNIST(download_root, train=False, download=True)
test_dataset = MNIST(download_root, train=False, download=True)