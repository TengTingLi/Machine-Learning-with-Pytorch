import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt

## prepare data
# get data
train_dataset = datasets.MNIST(root='', train=True)
test_dataset = datasets.MNIST(root='', train=False) # torch.size([60000, 28, 28])

# train data
x_train = train_dataset.data.reshape(-1, 28 * 28) # tensor([60000, 28^2])
x_train = x_train.float() / 255                   # normalize
y_train = train_dataset.targets                   # tensor([60000]) 

# test data
x_test = test_dataset.data.reshape(-1, 28 * 28) # tensor([10000, 28^2])
x_test = x_test.float() / 255                   # normalize
y_test = test_dataset.targets                   # tensor([10000]) 


## define model
class autoencoder(nn.Module):
    def __init__(self, compress_num):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, compress_num),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(compress_num, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# training loop defination
def training(model, loss_fn, optimizer, n_epoch):
    train_loss_hist = []
    test_loss_hist = []
    epoch_prog = tqdm.tqdm(range(n_epoch), unit='epoch', colour='GREEN')
    for epoch in epoch_prog:
        model.train()
        train_loss = []
        for image, label in train_load:
            pred = model(image)
            loss = loss_fn(pred, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss_hist.append(np.mean(train_loss[batch_size - 6:]))

        model.eval()
        torch.no_grad()
        pred = model(x_test)
        loss = loss_fn(pred, x_test)
        torch.enable_grad()
        test_loss_hist.append(loss.item())
        epoch_prog.set_postfix(train_loss = train_loss_hist[epoch], test_loss = test_loss_hist[epoch])
    
    return train_loss_hist, test_loss_hist

## training
# define parameter
n_epoch = 15
batch_size = 100
n_batch = len(x_train) // batch_size
train_load = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)
train_loss_hist, test_loss_hist = [], []


model_prog = tqdm.tqdm(range(9), unit='model')
for i in model_prog:
    model = autoencoder(784 // (10 * i + 10))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    temp_train , temp_test = training(model, loss_fn, optimizer, n_epoch)
    train_loss_hist.append(temp_train)
    test_loss_hist.append(temp_test)

## result
torch.save(model, 'Auto-encoder model on MNIST.pth')

# plot

main, sub = plt.subplots(3, 3)
for i in range(9):
    ro, co = i // 3, i % 3
    sub[ro, co].plot(train_loss_hist[i], label="train")
    sub[ro, co].plot(test_loss_hist[i], label="test")
    sub[ro, co].set_ylabel("Mean Square Error loss")
    sub[ro, co].legend()
plt.show()

'''model = torch.load('Auto-encoder model on MNIST.pth')
model.eval
pred = model.encoder(x_test)
pred = pred.detach().tolist() # size [10000, 8]
colour = y_test.tolist() # size [10000]

sorted = [[]] * 10 # [number, feature]

prog = tqdm.tqdm(range(10))
for n in prog:
    # sort to n number
    num = []
    for i in range(len(y_test)):
        if(y_test[i] == n):
            num.append(pred[i])
    # transpost
    num = np.array(num)
    num = num.T.tolist()
    sorted[n] = num
        

# plot feature 0, and feature 1
main, sub = plt.subplots(2,5, sharex=True, sharey=True)
for n in range(10):
    ro , co = n // 5, n % 5
    sub[ro, co].scatter(sorted[n][1], sorted[n][2])
    
plt.show()'''