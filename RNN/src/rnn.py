import torch
import torch.nn as nn
from torch.nn.modules import linear
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.rnn = nn.RNN(self.input_size, self.hidden_size)        
        self.linear1 = nn.Linear(2, 4)
    
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _states = self.rnn(x)

        x = self.linear1(x)
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    h = [1, 0, 0, 0]
    e = [0, 1, 0, 0]
    l = [0, 0, 1, 0]
    o = [0, 0, 0, 1]
    
    hidden_size = 2
    Epochs = 50

    x_data = torch.FloatTensor(np.array([[h, e, l, l, o]], dtype=np.float32)).to(device)

    model = RNN(input_size=4, hidden_size=2).to(device)
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(Epochs):
        model.train()
        output = model(x_data)

        optimizer.zero_grad()
        loss = criterion(output, x_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch : {epoch+1}, Loss : {loss.item()}")
        prediction = torch.argmax(output, dim=2)
        print(prediction)
