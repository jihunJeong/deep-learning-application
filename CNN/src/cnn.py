from os import access
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.optim as optim

from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchsummary import summary

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.linear1 = nn.Linear(64*7*7, 248)
        self.linear2 = nn.Linear(248, 10)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.xavier_normal_(m.bias.data, gain=0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.sigmoid(x)

        x = self.linear2(x)
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

    batch_size = 128
    lr = 0.001
    Epochs = 10
    n_input = 28*28
    n_hidden = 256

    train_dataset = MNIST(download_root, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(download_root, train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=True)
    total_batch = len(train_dataset)

    model = Cnn().to(device)
    summary(model, (1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(Epochs):
        model.train()
        avg_loss = 0
        for iter, batch in enumerate(train_loader):
            image, label = batch
            image, label = image.to(device), label.to(device)
            output = model(image)

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            avg_loss += loss / total_batch
        print(f"Epoch : {epoch+1}, Loss : {avg_loss}")

        model.eval()
        accuracy, total_test = 0, len(test_loader)
        for iter, batch in enumerate(test_loader):
            with torch.no_grad():   
                image, label = batch
                image, label = image.to(device), label.to(device)
                output = model(image)

                correct = (torch.argmax(output, dim=1) == label).float().mean()
                accuracy += correct / total_test
        print(f"Accuracy : {accuracy}")