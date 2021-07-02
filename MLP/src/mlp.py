from numpy.core.numeric import _correlate_dispatcher
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import pickle
import gzip

import numpy as np
import requests

from pathlib import Path
from matplotlib import pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.softmax(x)
        return x
        
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_PATH = Path("../data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)
    FILENAME = "mnist.pkl.gz"

    # download path 정의
    download_root = '../data'

    train_dataset = MNIST(download_root, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(download_root, train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)
    total_batch = len(train_loader)

    test_loader = DataLoader(dataset=test_dataset, batch_size=16, drop_last=True)
    Epochs = 30

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(Epochs):
        avg_loss = 0
        
        model.train()
        for iter, batch in enumerate(train_loader):
            train, label = batch
            train, label = train.to(device), label.to(device)
            output = model(train)
            print(output)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            correct = torch.argmax(output, 1) == label
            accuracy = correct.float().mean()
            avg_loss += loss / total_batch
        print(f"Epoch : {epoch+1}, Loss : {avg_loss}")

        accuracy, test = 0, 0

        model.eval()
        for iter, batch in enumerate(test_loader):
            with torch.no_grad():
                test, label = batch
                test, label = test.to(device), label.to(device)

                output = model(test)
                correct = (torch.argmax(output, 1) == label).float().mean()
                accuracy += correct
        accuracy = accuracy / len(test_loader)
        print(f"Accuracy : {accuracy}")
                    