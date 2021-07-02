import torch
import torch.nn as nn
import torch.optim as optim

from xor_mlp import Epochs

X = torch.FloatTensor([[0, 0],[0, 1],[1, 0],[1, 1]])
Y = torch.FloatTensor([[1, 0],[0, 1],[0, 1],[1, 0]])

w1 = torch.FloatTensor(2, 3).uniform_(-1, 1)
b1 = torch.FloatTensor(3).uniform_(-1, 1)
l1 = nn.Sigmoid(torch.matmul(X, w1)+b1)

w2 = torch.FloatTensor(3, 1).uniform_(-1, 1)
b2 = torch.FloatTensor(1).uniform_(-1, 1)
logits = torch.add(torch.matmul(l1, w2), b2)

output = nn.Sigmoid(logits)

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(lr=0.1)
Epochs = 8000
for epoch in range(Epochs):
    opt.zero_grad()
    loss = criterion(output, Y)
    loss.backward()
    opt.step()