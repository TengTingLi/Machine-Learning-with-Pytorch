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
train_dataset = datasets.MNIST(root='', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='', train=False, transform=transforms.ToTensor())

# load data in batch (cuz dun wan load all image at once, not memory freindly ig)
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class logistic_regression(nn.Module):
    def __init__(self, n_input, n_output):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        x = self.linear(x)
        return x
    
# initialize modal and parameters ig
n_input = 28 * 28
n_output = 10
model = logistic_regression(n_input, n_output)
model_load = torch.load("MNIST_trained_model.pth")

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  # SGD = stochastic gradiend descent
n_epoch = 50
ce_loss = [[], []]
acc = []

# play with model
model.eval()
torch.no_grad()

image_data = [None] * 32
label = [None] * 32

image_data, label = test_loader

for image_data, label in test_loader:
    pass


output = model_load(image_data[1].view(-1, 28*28))
print(output)
what , predicted = torch.max(output.data, 1)
print(what)
print(predicted)

'''
prog_bar = tqdm.tqdm(range(n_epoch), ncols=200, unit='bitch')
# training loop
for epoch in prog_bar:
    
    model.train()
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(image.view(-1, 28 * 28))
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
    ce_loss[0].append(loss.item())

    model.eval()
    torch.no_grad()
    correct = 0
    for image, label in test_loader:
        output = model(image.view(-1, 28 * 28))
        
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label).sum()
    acc.append(100 * correct.item() / len(test_dataset))
    prog_bar.set_postfix(train_ce_loss = ce_loss[0][epoch], test_acc = acc[epoch])

    torch.enable_grad()
    
# save model
torch.save(model, 'MNIST_trained_model.pth')



# plot
main, sub = plt.subplots(1,2, figsize=(9,4))
sub[0].plot(ce_loss[0], label="Cross Entropy loss")
sub[1].plot(acc, label="Accuracy")
sub[0].legend()
sub[1].legend()
sub[0].set_ylabel("Cross Entropy loss")
sub[1].set_ylabel("Accuracy")
plt.show()
'''