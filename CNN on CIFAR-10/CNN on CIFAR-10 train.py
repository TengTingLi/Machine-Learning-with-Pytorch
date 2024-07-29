import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# model class
class CNN_class(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(8192, 512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512,10)
        
    def forward(self, x):
        # print(x.size())
        # o = p

        # input 3x32x32 output 32x32x32
        x = self.drop1(self.act1(self.conv1(x)))

        # input 32x32x32 output 32x32x32
        x = self.pool2(self.act2(self.conv2(x)))

        # input 32x32x32 output 8192
        x = self.flat(x)

        # input 8192 ouptut 512
        x = self.drop3(self.act3(self.fc3(x)))

        # input 512 output 10
        x = self.fc4(x)

        return x
    
# load data make bactch
train_set = datasets.CIFAR10(root='.', train=True, transform=transforms.ToTensor())
test_set = datasets.CIFAR10(root='.', train=False, transform=transforms.ToTensor())
train_batch = DataLoader(train_set, batch_size=30, shuffle=True)
test_batch = DataLoader(test_set, batch_size=30, shuffle=True)

# initialize model and hyperparamaters
model_CNN = CNN_class()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_CNN.parameters(), lr=0.001, momentum=0.9)
n_epoch = 30
ce_loss_hist = [[],[]]
acc_hist = [[], []]


# training loop


for epoch in range(n_epoch):
    
    model_CNN.train()
    ce_loss, acc = [] , []
    prog_bar_train = tqdm.tqdm(train_batch, unit='batch', mininterval=1, desc=f"Train Epoch {epoch}")
    for train_input, train_label in prog_bar_train:
        '''print(train_input)
        print(train_input.size())
        print(model_CNN(train_input))
        print(model_CNN(train_input).size())
        print(train_label)
        print(train_label.size())
'''
        # op = gangnam

        
        #training mode
        pred = model_CNN(train_input)
        # print('pred\n',pred)
        # print('pred\n',train_label)
        '''plt.plot(pred)
        plt.show'''
        loss = loss_fn(pred, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # find train loss and acc in this epoch
        acc.append((torch.argmax(pred, 1) == train_label).float().mean().item()) # [mean]
    ce_loss_hist[0].append(loss.item())
    acc_hist[0].append(np.mean(acc))
    # print(f"train_ce_loss={ce_loss_hist[0][epoch]}, train_acc={acc_hist[0][epoch]}")
    

    model_CNN.eval()
    torch.no_grad()
    ce_loss, acc = [] , []
    prog_bar_test = tqdm.tqdm(test_batch, unit='batch', mininterval=1, desc=f"Test Epoch {epoch}")
    for test_input, test_label in prog_bar_test:
        pred = model_CNN(test_input)

        # find test loss
        ce_loss.append(loss_fn(pred, test_label).item())
        
        # find test acc
        acc.append((torch.argmax(pred, 1) == test_label).float().mean().item())

    ce_loss_hist[1].append(np.mean(ce_loss))
    acc_hist[1].append(np.mean(acc))
    # print(f"test_ce_loss={ce_loss_hist[1][epoch]}, test_acc={acc_hist[1][epoch]}")
    torch.enable_grad()

torch.save(model_CNN, 'model_CNN_on_CIFAR.pth')


# load model and see feature map
model_CNN = torch.load('model_CNN_on_CIFAR.pth')
n = 53
print(train_set[n][1])
plt.imshow(train_set.data[n])
plt.show()
x = torch.tensor([train_set.data[n]], dtype=torch.float32)
x = x.permute(0,3,1,2)
print(x.size())
model_CNN.eval()
with torch.no_grad():
    feature_map_1 = model_CNN.conv1(x)
    feature_map_2 = model_CNN.conv2(model_CNN.drop1(model_CNN.act1(feature_map_1)))
    pred = model_CNN(x)
print(torch.argmax(pred))

# display first convoluted layer
main, sub = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0,32):
    row, col = i//8, i%8
    sub[row][col].imshow(feature_map_1[0][i])
plt.show()

# display second convoluted layer
main, sub = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
for i in range(0,32):
    row, col = i//8, i%8
    sub[row][col].imshow(feature_map_2[0][i])
plt.show()


# plot

main, sub = plt.subplots(1,2,figsize=(10,5))
name = [["ce_loss", "acc"], ["train", "test"]]

sub[0].set_ylabel(name[0][0])
sub[0].set_xlabel("epoch")
sub[0].plot(ce_loss_hist[0], label=name[1][0])
sub[0].plot(ce_loss_hist[1], label=name[1][1])
sub[0].legend()
sub[1].set_ylabel(name[0][1])
sub[1].set_xlabel("epoch")
sub[1].plot(acc_hist[0], label=name[1][0])
sub[1].plot(acc_hist[1], label=name[1][1])
sub[1].legend()
'''plt.show()'''