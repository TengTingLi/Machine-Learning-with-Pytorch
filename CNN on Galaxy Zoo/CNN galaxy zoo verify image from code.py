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
    gz_class = ['Ec', 'Ei', 'Er']
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
        self.act = nn.Sequential(
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
            nn.Conv2d(128,174, kernel_size=(3,3), padding=1),
            # 12*12*256 -> 12*12*384 
            nn.Conv2d(174,174, kernel_size=(3,3), padding=1),
            # 12*12*384 -> 12*12*384
            nn.Conv2d(174,128, kernel_size=(3,3), padding=1),
            # 12*12*384 -> 12*12*256
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            # 12*12*256 -> 5*5*256
            nn.Flatten(),
            nn.Linear(5 * 5 * 128, 1096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1096, 1096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1096, n_class)
        )
        
    def forward(self, x):
        x = self.act(x)
        return x

def train(n_class, n_epoch, batch_size, path='CNN on gz2.pth', graph=False):
# parameter
    model = CNN(n_class)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9 ,weight_decay=0.9)
    bpe = len(x_train) // batch_size # batch per epoch
    loader_train = DataLoader(list(zip(x_train, y_train)), batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(list(zip(x_test, y_test)), batch_size=batch_size, shuffle=True)
    loss_hist = [[], []]
    acc_hist = [[], []]

## training
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
            main, sub = plt.subplots(3,5, figsize=(12,9))
            for j in range(len(train)):
                label_name = ['Ec', 'Ei', 'Er']
                row, col = j//5, j%5
                image_array = train[j].permute(1,2,0).numpy()
                sub[row][col].imshow(image_array)
                sub[row][col].axis('off')
                sub[row][col].set_title(label_name[real[j].argmax(0).item()])
            plt.show()

        # evaluate
        model.eval()
        loss_test = []
        acc_test = []
        epoch_best = []

    return loss_hist, acc_hist, loss_best, acc_best, epoch_best

loss_run_hist = []
acc_run_hist = []
loss_best_hist = []
acc_best_hist = []
epoch_best_hist = []
npc = [500]
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
    loss_temp, acc_temp, loss_best, acc_best, epoch_best = train(n_class, 200, 15, path=path, graph=True)
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