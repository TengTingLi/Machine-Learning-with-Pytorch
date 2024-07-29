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

## Prepare data
# load csv
print('loading csv...')
hart16 = pd.read_csv("gz2_hart16.csv")
column_needed = [0, 5, 6, 199, 205]
hart16 = hart16.iloc[:, column_needed]
hart16.rename(columns={'dr7objid' : 'objid'}, inplace=True)
mapping = pd.read_csv("gz2_filename_mapping.csv")
# validate if all file is avaible
image_filename = os.listdir('image')
image_filename = [os.path.splitext(filename)[0] for filename in image_filename] # remove '.jpg'
image_filename = [int(filename) for filename in image_filename]
mapping = mapping[mapping['asset_id'].isin(image_filename)]
# get the final table
id_to_filename = pd.merge(hart16, mapping, how='inner')
category = id_to_filename.columns[3:len(column_needed)].values
print(category)

# custom Dataset class to load jpg
class gz2_filtered(Dataset):
    def __init__(self, root_dir, filelist, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []
        for i in range(len(filelist)):
            self.image_path.append(os.path.join(root_dir, f'{filelist[i]}.jpg'))
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
            image = self.transform(image)
        return image

def get_x_y(npc):
    # filter arm class
    gz_class = ['Ec', 'Er']
    new_gz_class = ['1', '2', '3']
    filtered_list = pd.DataFrame(columns=id_to_filename.columns.values) # make empty df
    for i, str in enumerate(gz_class):
        # gat class
        arm = id_to_filename[id_to_filename['gz2_class'].str.contains(str)]
        # arm = arm[arm['gz2_class'].str.contains('d')]
        # arm = arm[arm['gz2_class'].str.contains(str)]
        # arm = arm[arm[category[1]] > 0.8]
        print('class: ',str, arm.shape, end="")
        # filter out class with (whatever) except for when adding merge class
        arm = arm[~arm['gz2_class'].str.contains('\(.\)')] if str != '\(m\)' else arm
        print(arm.shape, end='\n\n')
        # rename gz2 class name with my own
        # arm.loc[:, 'gz2_class'] = new_gz_class[i]
        # cut training data size
        arm = arm.iloc[0:npc]
        filtered_list = pd.merge(filtered_list, arm, how='outer')

    # get the file and transform to tensor format
    filelist = filtered_list.loc[:, 'asset_id']
    transform = transforms.Compose([
        transforms.CenterCrop((212,212)),
        transforms.ToTensor()
    ])
    ''' this is my attempt to load image, turns out .stack() use too much memory if num image is huge  
    data = []
    for i in tqdm(range(len(filelist)), desc='loading jpg', mininterval=1):
        img = Image.open(os.path.join('image', f'{filelist[i]}.jpg'))
        img = transform(img) # downsize and transfor to tensor
        data.append(img)
    print('stacking data...', end="")
    data = torch.stack(data)
    print('done !')'''
    # load image path using Dataset library
    data = gz2_filtered('image', filelist, transform)
    

    # get label into ohe
    label = filtered_list.loc[:, 'gz2_class':'gz2_class']
    ohe = OneHotEncoder(sparse_output=False).fit(label)
    print(ohe.categories_[0])
    n_class = len(ohe.categories_[0])
    label = ohe.transform(label)
    label = torch.tensor(label, dtype=torch.float32)
    return data, label, n_class



# split data to test train
# x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.7, shuffle=True)

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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_class)
        )
        
    def forward(self, x):
        x = self.convolution(x)
        print(x.size())
        x = self.flat(x)
        print(x.size())
        x = self.classify(x)
        print(x.size())
        return x

def train(n_class, n_epoch, batch_size, path='CNN on gz2.pth', graph=False):
# parameter
    model = CNN_small(n_class)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.9)
    bpe = len(x_train) // batch_size # batch per epoch
    loader_train = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(list(zip(x_test, y_test)), batch_size=batch_size, shuffle=True)
    loss_hist = [[], []]
    acc_hist = [[], []]
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

    bar_epoch = tqdm(range(n_epoch), colour='#7f7fff', mininterval=1, leave=False, desc='Epoch')
    loss_best = np.Inf
    acc_best = 0
    epoch_best = 0
    for epoch in bar_epoch:
        # train
        model.train()
        loss_train = []
        acc_train = []
        bar_train = tqdm(loader_train, mininterval=1, leave=False, desc='Batch training')
        for i, (train, real) in enumerate(bar_train):
            optimizer.zero_grad()
            pred = model(train)
            loss = loss_fn(pred, real.argmax(1))
            loss.backward()
            optimizer.step()

            loss_train.append(loss)
            acc_train.append((pred.argmax(1) == real.argmax(1)).float().mean())
        # get the average of lass 3 train loss
        loss_train = torch.stack(loss_train)
        loss_hist[0].append(loss_train[len(loss_train) - 3 : len(loss_train)].mean().item())
        # get the average of last 3 accuracy
        acc_train = torch.stack(acc_train)
        acc_hist[0].append(acc_train[len(acc_train) - 3 : len(acc_train)].mean().item())

        # evaluate
        model.eval()
        loss_test = []
        acc_test = []
        bar_test = tqdm(loader_test, mininterval=1, leave=False, desc='Batch testing')
        with torch.no_grad():
            # because dataloader loader from my custom dataset is list[tensor()...] instead of just tensor
            for i, (test, real) in enumerate(bar_test):
                pred = model(test)
                loss_test.append(loss_fn(pred, real).item())
                acc_test.append((pred.argmax(1) == real.argmax(1)).float().mean().item())
            lr_schedule.step(np.mean(loss_test))

        # get average of test loss and acc
        loss_hist[1].append(np.mean(loss_test))
        acc_hist[1].append(np.mean(acc_test))
        bar_epoch.set_postfix_str("cross entropy loss: train=%.3f test=%.3f, accuracy: train=%.3f test=%.3f" 
                                  % (loss_hist[0][epoch], loss_hist[1][epoch], acc_hist[0][epoch], acc_hist[1][epoch]))
        # get best for model, loss , acc 
        if loss_hist[1][epoch] < loss_best:
            # save model state dict
            torch.save(model.state_dict(), path)
            loss_best = loss_hist[1][epoch]
            epoch_best = epoch
        acc_best = acc_hist[1][epoch] if acc_hist[1][epoch] > acc_best else acc_best    

    # plot loss and acc of this run
    if graph == True:
        main, sub = plt.subplots(1, 2, figsize=(10,5))
        sub[0].plot(loss_hist[0], label='train')
        sub[0].plot(loss_hist[1], label='test')
        sub[1].plot(acc_hist[0], label='train')
        sub[1].plot(acc_hist[1], label='test')
        sub[0].set_ylabel("cross entropy loss")
        sub[1].set_ylabel("accuracy")
        sub[0].set_xlabel("epoch")
        sub[1].set_xlabel("epoch")
        sub[0].legend()
        sub[1].legend()
        main.suptitle('best cross entropy loss = %.3f at epoch %d and best accuracy = %.3f\nnumber of sample per class = %d, 150 batch size' % (loss_best, epoch_best, acc_best, npc[run]))
        plt.show()

    return loss_hist, acc_hist, loss_best, acc_best, epoch_best

loss_run_hist = []
acc_run_hist = []
loss_best_hist = []
acc_best_hist = []
epoch_best_hist = []
npc = [1000]
# npc = [(10 - 1) * 10 for i in range(6)]
bar_run = tqdm(range(len(npc)), mininterval=10, colour='#bdffbd', desc='Run')
for run in bar_run:
    # perpare data
    bar_run.set_description_str('preparing data...')
    x, y, n_class = get_x_y(npc[run])
    bar_run.set_description_str('train test spliting...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
    bar_run.set_description_str('cooking...')

    # file naming
    path = os.path.join('v3 model', f'CNN v2 on gz2 with {npc[run]} npc 100 epoch 100 batch only 2 arm no bulge center cropped.pth')
    path = os.path.join('v3 model', f'test.pth')

    # train and stats time
    loss_temp, acc_temp, loss_best, acc_best, epoch_best = train(n_class, 50, 128, path=path, graph=True)
    loss_run_hist.append(loss_temp)
    acc_run_hist.append(acc_temp)
    loss_best_hist.append(loss_best)
    acc_best_hist.append(acc_best)
    epoch_best_hist.append(epoch_best)
    bar_run.set_postfix_str('At %d sample per class, best cross entropy loss = %.3f at epoch %d and best accuracy = %.3f' 
          % (npc[run], loss_best, epoch_best, acc_best))


## Evaluate and stuff
# load model
# model = CNN()
# model.load_state_dict(torch.load("CNN on gz2.pth"))

sub_x, sub_y = 2, 3
main, sub = plt.subplots(sub_x, sub_y, figsize=(17,8), sharex=True)
# plot cross entropy loss
'''for run in range(len(npc)):
    row, col = run // sub_y, run % sub_y
    sub[row][col].plot(loss_run_hist[run][0], label='train')
    sub[row][col].plot(loss_run_hist[run][1], label='test')
    sub[row][col].legend()
    sub[row][col].set_title(f'Sample per class={npc[run]}\nBest:{loss_best_hist[run]} at {epoch_best_hist[run]} epoch')
    main.suptitle('Cross Entropy Loss for all run')
    main.supxlabel('Epoch')
    main.supylabel('Cross Entropy Loss')
plt.show()
# plot accuracy
main, sub = plt.subplots(sub_x, sub_y, figsize=(17,8), sharex=True)
for run in range(len(npc)):
    row, col = run // sub_y, run % sub_y
    sub[row][col].plot(acc_run_hist[run][0], label='train')
    sub[row][col].plot(acc_run_hist[run][1], label='test')
    sub[row][col].legend()
    sub[row][col].set_title(f'Sample per class={npc[run]}\nBest:{loss_best_hist[run]} at {epoch_best_hist[run]} epoch')
    main.suptitle('Accuracy for all run')
    main.supxlabel('Epoch')
    main.supylabel('Accuracy')
plt.show()'''