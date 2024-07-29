import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import numpy as np
import matplotlib.pyplot as plt

## prepare data
data = pd.read_csv('SDSS boss data_1.csv') # in 2d dataframe
x = data.iloc[:, 2:9]
y = data.iloc[:, 9:]
print(x.columns)

# create ohe encoder object
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
print('label = %s' % ohe.categories_)

y = ohe.transform(y)

x = torch.tensor(x.to_numpy(), dtype=torch.float32) # size = [19057 , 9]
y = torch.tensor(y, dtype=torch.float32)            # size = [19057 , 3]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

## define model
class classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Sequential(
            nn.Linear(7,18),
            nn.ReLU(),
            nn.Linear(18,9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )

    def forward(self, x):
        return self.act(x)

## define hyperparameter

train_time = 1
n_epoch = 50
batch = [10]
wd = [0.1]
train_loss_hist = []
test_loss_hist = []
train_acc_hist = []
test_acc_hist = []

# graph stuff
sub_row, sub_column = 5, 2
main_1 , sub_1 = plt.subplots(sub_row, sub_column, figsize=(7, 12), sharex=True)
main_2 , sub_2 = plt.subplots(sub_row, sub_column, figsize=(7, 12), sharex=True)
main_1.suptitle("BOSS object classification Cross Entropy Loss")
main_2.suptitle("BOSS object classification  Accuracy")
main_1.supylabel("Cross Entropy loss")
main_2.supylabel("Accuracy")
main_1.supxlabel("Epoch")
main_2.supxlabel("Epoch")

## training loop
for n in tqdm.trange(train_time, colour='#0077ff'):
    # model initialazation
    model = classification()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_loss = 1000
    best_acc = 0
    row, col = n // sub_column, n % sub_column

    prog = tqdm.tqdm(range(n_epoch), unit='epoch', desc=f'train {n}', leave=False)
    for epoch in prog:
        model.train()
        loss_5 = []
        acc_5 = []
        for i in range(len(x_train) // batch[n]):
            y_pred = model(x_train[i : i + batch[n]])
            loss = loss_fn(y_pred, y_train[i : i + batch[n]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_5.append(loss.item())
            acc_5.append((y_pred.argmax(1) == y_train[i : i + batch[n]].argmax(1)).float().mean().item())
        
        train_loss.append(np.mean(loss_5[4:])) # size = [epoch]
        train_acc.append(np.mean(acc_5[4:])) # size = [epoch]

        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
            loss = loss_fn(y_pred, y_test)
            test_loss.append(loss.item()) # size = [epoch]
            test_acc.append((y_pred.argmax(1) == y_test.argmax(1)).float().mean().item())
            if best_loss >= test_loss[epoch] : best_loss = test_loss[epoch]
            if best_acc <= test_acc[epoch] : best_acc = test_acc[epoch]

        prog.set_postfix(entropy_loss = [train_acc[epoch], test_acc[epoch]])

    # plotting n stuff
    sub_1[row][col].plot(train_loss, label="train")
    sub_1[row][col].plot(test_loss, label="test")
    sub_1[row][col].legend()
    sub_1[row][col].set_title("Batch size = %dbest loss =%.3f" %(batch[n] , best_loss))
    sub_2[row][col].set_title("Batch size = %dbest acc =%.3f" %(batch[n] , best_acc))
    sub_2[row][col].plot(train_acc, label="train")
    sub_2[row][col].plot(test_acc, label="test")
    sub_2[row][col].legend()
plt.show()
    


## save and result
# torch.save(model, 'classification on SDSS Boss object type.pth')

check = [0] * 9
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    y_pred = y_pred.argmax(1) 
    type = y_test.argmax(1) 
    bar_check= tqdm.trange(len(x_test))
    for i in bar_check:
        for j in range(3):
            for k in range(3):
                if type[i] == j: 
                    if y_pred[i] == k: check[j*k] += 1
                bar_check.set_postfix(c=check)

# see model in action
for i in range(5):
    pred = model(x_test[i])
    print(f' predicted {pred.detach().numpy()}, actual {y_test[i].detach().numpy()}')

