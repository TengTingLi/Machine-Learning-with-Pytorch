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

# custom Dataset class to load jpg
class gz2_filtered(Dataset):
    def __init__(self, npc, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_path = []
        self.label = []
        # get list of dir
        class_dir = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        # make a dic for dir list to class
        self.class_to_label = {class_dir: i for i, class_dir in enumerate(class_dir)}
        print(self.class_to_label, end=' ')

        for class_temp in class_dir:
            class_path = os.path.join(root_dir, class_temp)
            for i, image_name in enumerate(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_name)
                self.image_path.append(image_path)
                self.label.append(self.class_to_label[class_temp])
                if i == (npc - 1): break
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        if self.transform:
            image = self.transform(image)
        
        label = self.label[idx]
        return image, label



## Define model and parameters
class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1),
            # 212*212*3 -> 212*212*16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            # 212*212*16 -> 106*106*16

            nn.Conv2d(32, 16, kernel_size=(3,3), stride=1, padding=1),
            # 106*106*16 -> 106*106*8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # 106*106*8 -> 53*53*8

            nn.Conv2d(16, 8, kernel_size=(3,3), stride=1, padding=1),
            # 53*53*8 -> 53*53*8 
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
            # 53*53*8 -> 26*26*8 (26.5 -> 26)
        )
        self.flat = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(5408,5408),
            nn.ReLU()
        )
        self.decoder = nn.Sequential( 
            nn.Upsample(size=53),
            # 26*26*8 -> 53*53*8
            nn.ConvTranspose2d(8, 16, kernel_size=(3,3), stride=1, padding=1, output_padding=0),
            # 53*53*8 -> 53*53*8
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            # 53*53*8 -> 106*106*8
            nn.ConvTranspose2d(16, 32, kernel_size=(3,3), stride=1, padding=1, output_padding=0),
            # 106*106*8 -> 106*106*16
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            # 106*106*16 -> 212*212*16
            nn.ConvTranspose2d(32, 3, kernel_size=(3,3), stride=1, padding=1, output_padding=0),
            # 212*212*16 -> 212*212*3
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(self.flat(x))
        x = x.reshape((-1, 8, 26, 26))
        x = self.decoder(x)
        return x

# perpare data
npc = 20
transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # load image path using Dataset library

print('get data...', end="")
data = gz2_filtered(npc, 'image sort', transform)
print("total image:", data.__len__(), end=" ")
print('done')

n_class = 3
n_epoch = 50
batch_size = 50
model = CAE()
model.load_state_dict(torch.load("Autoencoder model\\autoencoder test.pth"))
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
data_train, data_test = train_test_split(data, train_size=0.75, shuffle=False) 
loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)
loss_train_hist = []
loss_test_hist = []

bar_epoch = tqdm(range(n_epoch), colour='#7f7fff', mininterval=1, leave=False, desc='Epoch')
for epoch in bar_epoch:
    # train
    '''model.train()
    
    bar_train = tqdm(loader_train, mininterval=1, leave=False, desc='Batch training')
    for i, (train, real) in enumerate(bar_train):
        
        optimizer.zero_grad()
        pred = model(train)
        loss_train = loss_fn(pred, train)
        loss_train.backward()
        optimizer.step()'''
        

    model.eval()
    with torch.no_grad():
        bar_test = tqdm(loader_train, mininterval=1, leave=False, desc='Batch training')
        for i, (test, real) in enumerate(bar_test):
            '''pred = model(test)
            loss_test = loss_fn(pred, test)'''
            dic = {i : file_class for i, file_class in enumerate(os.listdir("image sort"))}
            print(dic)
            image = []
            pred_image = []
            for j, img in enumerate(test):
        
                
                img = img.reshape(-1, 3, 212, 212)
                
                pred_img = model(img)
                
                pred_img = pred_img.reshape(3, 212, 212)
                img = img.reshape(3, 212, 212)

                image.append(img.permute(1,2,0).numpy())
                pred_image.append(pred_img.permute(1,2,0).numpy())
                
            main, sub = plt.subplots(3,6, figsize=(10,8))
            for j in range(9):
                row, col = j // 3, j % 3
                sub[row][col * 2].imshow(image[j])
                sub[row][col * 2].set_title(f'Original {dic[real[j].item()]}')
                sub[row][col * 2].axis('off')
                sub[row][col * 2 + 1].imshow(pred_image[j])
                sub[row][col * 2 + 1].set_title("Autoencode")
                sub[row][col * 2 + 1].axis('off')
                main.suptitle(real[j])
            plt.show()
    
'''    loss_train_hist.append(loss_train.item())
    loss_test_hist.append(loss_test.item())
    bar_epoch.set_postfix_str(f'loss: train={loss_train.item()} test={loss_test.item()}')


torch.save(model.state_dict(), "Autoencoder model\\autoencoder test.pth")
plt.plot(loss_train_hist, label='train')
plt.plot(loss_test_hist, label='test')
plt.title("loss history of Autoencode")
plt.xlabel("mean square error loss")
plt.ylabel("epoch")
plt.legend()
plt.show()'''

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