import torch
import torch.nn as nn
import numpy as np
from torch.nn import modules
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torchvision.transforms as transforms

from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(AutoEncoder, self).__init__()

        self.flatten = nn.Flatten()

        self.encode = nn.Linear(n_input, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.decode = nn.Linear(n_hidden, n_input)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)

        x = self.encode(x)
        x = self.sigmoid(x)
        
        x = self.hidden(x)
        x = self.sigmoid(x)
        
        x = self.decode(x)
        x = x.view(128,1,28,28)
        
        return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_PATH = Path("../data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)
    FILENAME = "mnist.pkl.gz"

    # download path 정의
    download_root = '../data'

    batch_size = 128
    lr = 0.001
    Epochs = 10
    n_input = 28*28
    n_hidden = 256

    train_dataset = MNIST(download_root, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(download_root, train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_batch = len(train_loader)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=True)
    test_batch =  len(test_loader)

    model = AutoEncoder(n_input, n_hidden).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    model.train()
    for epoch in range(Epochs):
        avg_loss = 0
        for iter, batch in enumerate(train_loader):
            image, label = batch
            image, label = image.to(device), label.to(device)
            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, image)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss / total_batch
        print(f"Epoch : {epoch+1}, Loss : {avg_loss}")
        if epoch == 8:
            out_img = torch.squeeze(output.cpu().data)
            print(out_img.size())
            plt.imshow(out_img[0].numpy(),cmap='gray')
            plt.show()
