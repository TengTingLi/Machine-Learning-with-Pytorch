import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt

# get data from idk
train_dataset = datasets.MNIST(root='', train=True, download=True)
test_dataset = datasets.MNIST(root='', train=False)
print(train_dataset.data.size()) # torch.size([60000, 28, 28])

x_train = train_dataset.data.reshape(-1,784) # torch.size([60000, 28^2])
x_train = x_train.float() / 255.0 # normalize
y_train = train_dataset.targets # torch.size([60000])

x_test = test_dataset.data.reshape(-1,784) # torch.size([60000, 28^2])
x_test = x_test.float() / 255.0 # normalize
y_test = test_dataset.targets # torch.size([60000])


import PIL.Image as img
import numpy as np
# how many image you want to generate
num_of_image = 10

for i in range(num_of_image):
    # look at array
    '''print(x_test.reshape(-1, 28*28)[i])'''
    # convert to np.array
    starting_punya_index_in_MINST = 3 
    export_txt = x_test[i + starting_punya_index_in_MINST].reshape(28*28).numpy()

    '''
    # check image show number
    plt.imshow(export_txt.reshape(28,28), cmap='gray')
    plt.show()'''

    # just file name with the correct classification
    what_number_is_this = y_test[i + starting_punya_index_in_MINST].numpy()
    name_txt = str(what_number_is_this) + ".txt"
    name_img = str(what_number_is_this) + ".png"

    # export to txt
    
    np.savetxt(name_txt, export_txt)
    '''
    plt.imshow(np.loadtxt(name_txt[i]).reshape(28,28), cmap='gray')
    plt.show()'''

    # convert to png
    export_img = export_txt.reshape(28, 28)
    export_img = export_img * 255

    print(export_img)
    image = img.fromarray(np.uint8(export_img), mode='L')


    # export to png
    image.save(name_img)
    print("next loop")



# ignore code below





'''# size after kernaling
def sak(size, k, s, p):
    return int((size - k + 2 * p) // s) + 1

class CNN(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.CNN_1 = nn.Conv2d(1, out_channel, kernel_size=3, stride=1, padding=2)
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer_2 = nn.Linear(14*14*out_channel, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.CNN_1(x)
        x = self.act_1(x)
        x = self.pool_1(x)
        x = nn.Flatten()(x)
        x = self.layer_2(x)
        return x'''




'''# play with model

main_1 , sub_1 = plt.subplots(4,4, figsize=(12, 10), sharex=True)
plt.subplots_adjust(hspace=0.4)
main_2 , sub_2 = plt.subplots(4,4, figsize=(12, 10), sharex=True)
plt.subplots_adjust(hspace=0.4)
main_1.suptitle("Grid Search Cross Entropy Loss")
main_2.suptitle("Grid Search Accuracy")
main_1.supylabel("Cross Entropy loss")
main_2.supylabel("Accuracy")
main_1.supxlabel("Epoch")
main_2.supxlabel("Epoch")
lr = [0.1 ** i for i in range(4)]
out_channel = [1 + i for i in range(4)]

# training loop
bar_lr = tqdm.tqdm(range(len(lr)), colour="#ffffff", position=0)
for run_lr in bar_lr:
    bar_lr.set_description_str("lr = %.3f" % lr[run_lr]) 
    bar_channel = tqdm.tqdm(range(len(out_channel)), colour="#ffbdbd", leave=False, position=1)
    for run_channel in bar_channel:
        bar_channel.set_description_str("channel amount = %.3f" % out_channel[run_channel]) 

        n_epoch = 50
        batch_size = 100
        model = CNN(out_channel[run_channel])
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr[run_lr])
        train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)
        ce_loss = [[], []]
        acc = [[], []]
        ce_best = 1000
        acc_best = 0

        prog_bar = tqdm.tqdm(range(n_epoch), colour='#ff7e7e', leave=False, position=2)
        for epoch in prog_bar:
            
            model.train()
            # bar_train = tqdm.tqdm(train_loader, leave=False, colour='#FF3F3F', mininterval=1, position=3)
            # for i, (image, label) in enumerate(bar_train): # [[batch 1], [batch 2], ...]
            for i, (image, label) in enumerate(train_loader): # [[batch 1], [batch 2], ...]
                
                optimizer.zero_grad()
                output = model(image)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
            ce_loss[0].append(loss.item())
            acc[0].append((torch.argmax(output, 1) == label).float().mean().item())

            # validation
            model.eval()
            with torch.no_grad():
                output = model(x_test) # torch.size([10000, 10])
                loss = loss_fn(output, y_test)
                ce_loss[1].append(loss.item())
                acc_test = (torch.argmax(output, 1) == y_test).float().mean().item()
                acc[1].append(acc_test)
                if ce_loss[1][epoch] <= ce_best: ce_best = ce_loss[1][epoch]
                if acc[1][epoch] >= acc_best: acc_best = acc[1][epoch]

            prog_bar.set_postfix(loss_train = ce_loss[0][epoch], loss_test = ce_loss[1][epoch], acc_train = acc[0][epoch], acc_test= acc[1][epoch])

        # plotting n stuff
        sub_1[run_lr][run_channel].plot(ce_loss[0], label="train")
        sub_1[run_lr][run_channel].plot(ce_loss[1], label="test")
        sub_1[run_lr][run_channel].legend()
        sub_1[run_lr][run_channel].set_title("lr=%.3f, channel=%d,\nbest loss =%.3f" %(lr[run_lr], out_channel[run_channel], ce_best))
        sub_2[run_lr][run_channel].set_title("lr=%.3f, channel=%d,\nbest acc =%.3f" %(lr[run_lr], out_channel[run_channel], acc_best))
        sub_2[run_lr][run_channel].plot(acc[0], label="train")
        sub_2[run_lr][run_channel].plot(acc[1], label="test")
        sub_2[run_lr][run_channel].legend()
plt.subplot_tool()
plt.show()'''

# save model
# torch.save(model, 'MNIST_trained_model.pth')

'''import PIL.Image as img
model = torch.load('MNIST_trained_model.pth')
model.eval()
with torch.no_grad():
    bar_see = tqdm.trange(len(y_test))
    for i in bar_see:
        pred = model(x_test)
        if pred.argmax(1)[i] != y_test[i]:
            plt.imshow(x_test.reshape(-1,28,28)[i], cmap='gray')
            plt.title("predicted %d, actual is %d" % (pred[i].argmax().item(), y_test[i].item()))
            plt.axis('off')
            plt.show()'''


'''import PIL.Image as img
model = torch.load('MNIST_trained_model.pth')
model.eval()
with torch.no_grad():
    bar_see = tqdm.trange(len(y_test))
    for i in bar_see:
        pred = model(x_test)
        if pred.argmax(1)[i] != y_test[i]:
            plt.imshow(x_test.reshape(-1,28,28)[i], cmap='gray')
            plt.title("predicted %d, actual is %d" % (pred[i].argmax().item(), y_test[i].item()))
            plt.axis('off')
            plt.show()'''

'''# plot
main, sub = plt.subplots(1,2, figsize=(9,4))
sub[0].plot(ce_loss[0], label="train")
sub[0].plot(ce_loss[1], label="test")
sub[1].plot(acc[0], label="train")
sub[1].plot(acc[1], label="test")
sub[0].legend()
sub[1].legend()
sub[0].set_ylabel("Cross Entropy loss")
sub[1].set_ylabel("Accuracy")
plt.show()'''
