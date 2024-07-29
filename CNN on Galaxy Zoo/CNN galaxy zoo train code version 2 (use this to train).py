import pandas as pd
import os
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import numpy as np
import copy

## Prepare data

# custom Dataset class to load jpg
class gz2_filtered(Dataset):
    def __init__(self, npc, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.ToTensor()
        self.image_path = []
        self.label = []
        # get list of dir
        class_dir = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # make a dic for dir list to class
        self.class_to_label = {class_dir: i for i, class_dir in enumerate(class_dir)}
        # this class is always inside run loop so it's save to do this
        bar_run.set_postfix_str(class_dir)

        for class_temp in class_dir:
            class_path = os.path.join(root_dir, class_temp)
            for i, image_name in enumerate(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_name)
                self.image_path.append(image_path)
                self.label.append(self.class_to_label[class_temp])
                if i == npc: break
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
            image = self.transform(image)
        
        label = self.label[idx]
        return image, label



## Define model and parameters
class CNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11, 11), stride=4),
            # 212*212*3 -> 51*51*96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            # 51*51*96 -> 25*25*96
            nn.Conv2d(48, 128, kernel_size=(5,5), stride=1, padding=2),
            # 25*25*96 -> 25*25*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 25*25*256 -> 12*12*256
            nn.Conv2d(128,192, kernel_size=(3,3), padding=1),
            # 12*12*256 -> 12*12*384 
            nn.Conv2d(192,192, kernel_size=(3,3), padding=1),
            # 12*12*384 -> 12*12*384
            nn.Conv2d(192,128, kernel_size=(3,3), padding=1),
            # 12*12*384 -> 12*12*256
            nn.MaxPool2d(kernel_size=(3,3), stride=2)
            # 12*12*256 -> 5*5*256
        )
        self.flat = nn.Flatten()
        self.classify = nn.Sequential( 
            nn.Linear(5 * 5 * 128, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, n_class)
        )
        
    def forward(self, x):
        x = self.convolution(x)
        x = self.flat(x)
        x = self.classify(x)
        return x

class CNN_small(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            # 212*212*3 -> 212*212*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # 212*212*16 -> 106*106*16

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            # 106*106*16 -> 106*106*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # 106*106*8 -> 53*53*8

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # 53*53*8 -> 53*53*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # 53*53*8 -> 26*26*8
        )
        self.flat = nn.Flatten()
        
        self.classify = nn.Sequential( 
            nn.Linear(26 * 26 * 8, 256),
            nn.ReLU(),
            # nn.Dropout(0.99),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.99),
            nn.Linear(128, n_class)
        )
        
    def forward(self, x):
        x = self.convolution(x)
        # print(x.size())
        x = self.flat(x)
        # print(x.size())
        x = self.classify(x)
        # print(x.size())
        return x





# setting
thing_to_sort = 'bar 2 arm'
title = f'Classification for tightness of 2 arm spiral Galaxy'
tried = '3'
path = os.path.join('image sort', thing_to_sort)
n_run = 4

# storable variable for each run
run_best = 0
loss_run_hist = []
loss_best_hist = []
acc_run_hist = []
acc_best_hist = []

bar_run = tqdm(range(n_run), colour='#7fff7f')
for run in bar_run:
    # hyperparameter
    npc = 3500
    data = gz2_filtered(npc, path)
    n_class = len(data.class_to_label) # count num of class
    n_epoch = 75
    batch_size = 128
    model = CNN_small(n_class)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
    
    # storable variable for each epoch
    loss_epoch_hist = [[], []]
    acc_epoch_hist = [[], []]
    loss_best = np.inf
    acc_best = 0

    # Prepare data
    # load image path using Dataset library
    bar_run.set_description_str('getting data...')
    data = gz2_filtered(npc, path)

    bar_run.set_description_str('spliting data...')
    data_train, data_test = train_test_split(data, train_size=0.7, shuffle=True)

    bar_run.set_description_str('loadering data...')
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

    bar_run.set_description_str('cooking...')
    bar_epoch = tqdm(range(n_epoch), colour='#7f7fff', mininterval=1, leave=True, desc='Epoch')
    for epoch in bar_epoch:
        # train
        model.train()
        loss_train = []
        acc_train = []
        bar_train = tqdm(loader_train, mininterval=0.1, leave=False, desc='Batch training')
        for i, (train, label_train) in enumerate(bar_train):
            optimizer.zero_grad()
            pred = model(train)
            loss_train_temp = loss_fn(pred, label_train)
            loss_train_temp.backward()
            optimizer.step()

            # recored for later averaging
            loss_train.append(loss_train_temp.item())
            acc_train.append((torch.argmax(pred, 1) == label_train).float().mean().item())
            bar_train.set_postfix_str(f'loss train = {np.mean(loss_train)}, acc train = {np.mean(acc_train)}')

        model.eval()
        with torch.no_grad():
            # gonna have to append the test stat cause no good way to deal with dataloader
            loss_test = []
            acc_test = []
            for i, (test, label_test) in enumerate(loader_test):
                pred = model(test)
                loss_test.append(loss_fn(pred, label_test).item())
                acc_test.append((pred.argmax(1) == label_test).float().mean().item())
            
            
            # average the test stat smh
            loss_train = np.mean(loss_train)
            acc_train = np.mean(acc_train)
            # average the test stat smh
            loss_test = np.mean(loss_test)
            acc_test = np.mean(acc_test)
        
        # record epoch history
        loss_epoch_hist[0].append(loss_train.item())
        loss_epoch_hist[1].append(loss_test.item())
        acc_epoch_hist[0].append(acc_train)
        acc_epoch_hist[1].append(acc_test)
        bar_epoch.set_postfix_str('loss: train = %.4f, test = %.4f, acc: train = %.4f, test = %.4f' % (loss_train, loss_test, acc_train, acc_test))
        if loss_epoch_hist[1][epoch] < loss_best: loss_best = loss_epoch_hist[1][epoch]
        if acc_epoch_hist[1][epoch] > acc_best: 
            acc_best = acc_epoch_hist[1][epoch]
            # copy the best model parameter
            model_best_parameter = copy.deepcopy(model.state_dict())

        # early stopping if overfit
        if acc_train - acc_test > 0.3: 
            bar_run.set_postfix_str("early stopping due to overfitting")
            break

    # record run history
    loss_run_hist.append(loss_epoch_hist)
    loss_best_hist.append(loss_best)
    acc_run_hist.append(acc_epoch_hist)
    acc_best_hist.append(acc_best)

    # save the best model parameter for the whole run
    torch.save(model_best_parameter, os.path.join('result', thing_to_sort, f'{title} try {tried} run {run}.pth')) 
    




# todo list:
'''
[/][v] best stats
[/][v] smooth train stat
[ ][ ] prog bar for testing
[/][v] run loop
[/][v] move desc to bar
[/][v] move hyperparameter inside run loop
[/][ ] save best model for best run and  best epoch as well
[ ][ ] check classificaltion error
'''

## Evaluate and stuff
# load model
# model = CNN()
# model.load_state_dict(torch.load("CNN on gz2.pth"))

# plot loss and acc side by side
if n_run == 1:
    sub_x, sub_y = 1, 2
    main, sub = plt.subplots(sub_x, sub_y, figsize=(17,8), sharex=True)
    main.suptitle(title)
    sub[0].plot(loss_epoch_hist[0], label='train')
    sub[0].plot(loss_epoch_hist[1], label='test')
    sub[0].legend()
    sub[0].set_xlabel("Epoch")
    sub[0].set_ylabel("loss")
    sub[0].set_title("Cross Entropy Loss")
    sub[1].plot(acc_epoch_hist[0], label='train')
    sub[1].plot(acc_epoch_hist[1], label='test')
    sub[1].legend()
    sub[1].set_xlabel("Epoch")
    sub[1].set_ylabel("accuracy")
    sub[1].set_title("Accuracy")
    plt.show()

# plot whatever run data
if n_run > 1:
    sub_x, sub_y = 2, 2
    main, sub = plt.subplots(sub_x, sub_y, figsize=(17,8), sharex=True)
    # plot cross entropy loss
    for run in range(n_run):
        row, col = run // sub_y, run % sub_y
        sub[row][col].plot(loss_run_hist[run][0], label='train')
        sub[row][col].plot(loss_run_hist[run][1], label='test')
        sub[row][col].legend()
        sub[row][col].set_title('Run %d Best loss: %.4f' % (run, loss_best_hist[run]))
        main.suptitle('%s\nCross Entropy Loss' % title)
        main.supxlabel('Epoch')
        main.supylabel('Cross Entropy Loss')
    plt.show()
    # plot accuracy
    main, sub = plt.subplots(sub_x, sub_y, figsize=(17,8), sharex=True)
    for run in range(n_run):
        row, col = run // sub_y, run % sub_y
        sub[row][col].plot(acc_run_hist[run][0], label='train')
        sub[row][col].plot(acc_run_hist[run][1], label='test')
        sub[row][col].legend()
        sub[row][col].set_title('Run %d Best acc: %.4f' % (run, acc_best_hist[run]))
        main.suptitle('%s\nAccuracy' % title)
        main.supxlabel('Epoch')
        main.supylabel('Accuracy')
    plt.show()