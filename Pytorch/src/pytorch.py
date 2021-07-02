import torch
import numpy as np


nums = torch.arange(9)
ten = torch.tensor(10)
nine = torch.tensor(9)

print(ten)
print(type(ten))
print(torch.add(ten, nine))


shape = (2,3)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
print(ones)
print(zeros)

print(zeros.dtype)
print(zeros.device)
