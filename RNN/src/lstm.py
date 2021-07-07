import torch
import numpy as np


from torch._C import device
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cls):
        super(LSTM, self).__init__()

        self.input = input_size
        self.hidden = hidden_size

        self.lstm = nn.LSTM(self.input, self.hidden, num_layers=1, batch_first=True)
        
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(self.hidden, cls)

    def forward(self, x):
        x, _states = self.lstm(x)
        
        x = self.linear1(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    sample = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."
    idx2char = list(set(sample))
    char2idx = {c : i for i, c in enumerate(idx2char)}

    dic_size = len(char2idx)
    num_classes = len(char2idx)
    hidden_size = 15
    batch_size = 1
    sequence_length = len(sample)-1
    lr = 0.1
    Epochs = 50

    sample_idx = [char2idx[c] for c in sample]

    x_data = [[[0 for _ in range(len(idx2char))] for _ in range(sequence_length)]]
    for i, c in enumerate(sample_idx[:-1]):
        x_data[0][i][c] = 1
    x_data = torch.Tensor(x_data).to(device)
    
    y_data = [sample_idx[1:]]
    y_data = torch.Tensor(y_data).to(device)

    model = LSTM(dic_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(Epochs):
        output = model(x_data)

        optimizer.zero_grad()
        loss = criterion(output.view(-1, dic_size), y_data.view(-1).long())
        loss.backward()
        optimizer.step()
        
        print(f"Epoch : {epoch+1}, Loss : {loss.item()}")

        predict = torch.argmax(output, dim=2)
        result_str = [idx2char[c] for c in predict[0]]
        print(''.join(result_str))