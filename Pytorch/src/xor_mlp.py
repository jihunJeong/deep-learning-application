import torch
from torch import optim
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 2)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x


X = torch.FloatTensor([[0, 0],[0, 1],[1, 0],[1, 1]])
Y = torch.LongTensor([0, 1, 1, 0])


model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

Epochs = 30000
for epoch in range(Epochs):
    output = model(X)
    loss = criterion(output, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(epoch, loss.item())