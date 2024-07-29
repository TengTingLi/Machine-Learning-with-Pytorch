import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import matplotlib.pyplot as plt

# get data
data = pd.read_csv("iris.csv", header = None)


# slice data
x = data.iloc[:, 0:4]
y = data.iloc[:, 4:]

# make an one hot vector enconder object 
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
print(ohe.categories_)

# transform y to one hot vector
y = ohe.transform(y)

# convert x & y in to pytorch tensor
x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32) 

# split dataset in to train-test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

# make a multiclass class
class multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4,8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8,3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
    
# create model, loss function & optimizer
model = torch.load("iris-model.pth")


# test the model in action
model.eval()
with torch.no_grad():    
    for i in range(len(x_test)):
        x_sample = x_test[i:i+1]
        y_pred = model(x_sample)
        print(f"{x_sample.numpy()} -> {y_pred[0].numpy()} (right awnwers is {y_test[i].numpy()})")

acc = float((torch.argmax(y_pred, 1) == (torch.argmax(y_test)).float().mean()))
print(f"acc = {acc}")

