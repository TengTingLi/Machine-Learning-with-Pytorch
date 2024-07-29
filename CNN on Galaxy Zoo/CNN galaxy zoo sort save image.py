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
image_filename = [int(filename) for filename in image_filename] # covert string to int
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
    gz_class = ['1', '2', '3', '4']
    # gz_class = ['4l', '4m', '4t']
    new_gz_class = ['SB1', 'SB2', 'SB3', 'SB4']
    # new_gz_class = ['SB4l', 'SB4m', 'SB4t']
    filtered_list = pd.DataFrame(columns=id_to_filename.columns.values) # make empty df
    for i, str in enumerate(gz_class):
        # gat class
        arm = id_to_filename[id_to_filename['gz2_class'].str.contains('SB')]
        # arm = arm[arm['gz2_class'].str.contains('d')]
        arm = arm[arm['gz2_class'].str.contains(str)]
        # arm = arm[arm[category[0]] > 0.8]
        '''print(arm.to_string())
        print('class: ',str, arm.shape, end="")'''
        # filter out class with (whatever) except for when adding merge class
        arm = arm[~arm['gz2_class'].str.contains('\(.\)')] if str != '\(m\)' else arm
        print(arm.shape, end='\n\n')
        # print(arm.to_string())
        # rename gz2 class name with my own
        arm.loc[:, 'gz2_class'] = new_gz_class[i]
        # cut training data size
        # arm = arm.iloc[0:npc]
        filtered_list = pd.merge(filtered_list, arm, how='outer')

    # get the file and transform to tensor format
    filelist = filtered_list.loc[:, 'asset_id']
    transform = transforms.Compose([
        transforms.CenterCrop(212),
        transforms.ToTensor()
    ])

    # load image path using Dataset library
    data = gz2_filtered('image', filelist, transform)
    

    # get label into ohe
    print('ohe...', end='')
    label = filtered_list.loc[:, 'gz2_class':'gz2_class']
    ohe = OneHotEncoder(sparse_output=False).fit(label)
    print(ohe.categories_[0])
    class_n = len(ohe.categories_[0])
    class_label = ohe.categories_[0]
    label = ohe.transform(label)
    label = torch.tensor(label, dtype=torch.float32)
    print('finished')
    return data, label, class_n, class_label 

    
    

npc = [500]
path = 'bar x arm'

# folder path
sort_image, sort_label, class_n, class_label= get_x_y(npc[0])
for str in class_label:
    if os.path.isdir(os.path.join(path, str)) == False:
        os.makedirs(os.path.join(path, str))


print('loader loading...', end="")
loader_train = DataLoader(list(zip(sort_image, sort_label)), batch_size=100, shuffle=True)
print('finished')

# sort loop
bar_sort = tqdm(loader_train, mininterval=1, leave=True, desc='Batch training')
for i, (image_batch, real) in enumerate(bar_sort):
    for j, pixel in enumerate(image_batch):  
        img = (pixel.permute(1,2,0) * 255).byte().numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(path, class_label[real.argmax(1)[j].item()], f'{i * len(image_batch) + j}.jpg'))

