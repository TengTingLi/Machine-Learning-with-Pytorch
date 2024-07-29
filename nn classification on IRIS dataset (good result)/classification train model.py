import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import copy
import matplotlib.pyplot as plt

# get data
data = pd.read_csv("iris.csv", header = None)


# slice data
x = data.iloc[:, 0:4]
y = data.iloc[:, 4:]

# make an one hot vector enconder object 
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
print(ohe.categories_)

# transform y to one hot vector
y = ohe.transform(y)

# convert x & y in to pytorch tensor
x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32) 


# split dataset in to train-test

'''
# make a multiclass class
class multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4,8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8,3)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x
'''
main_1, sub_1 = plt.subplots(2, 3, figsize=(12,9), sharex=True, sharey=True)
main_2, sub_2 = plt.subplots(2, 3, figsize=(12,9), sharex=True, sharey=True)
main_1.suptitle("Cross Entropy Loss history for Iris species classification\nwith different learning rate")
main_2.suptitle("Accuracy history for Iris species classification\nwith different learning rate")
main_1.supxlabel('Epoch')
main_1.supylabel('Cross Entropy Loss')

lr = [0.1 ** (1+i) for i in range(6)]
# lr = [0.01 for i in range(6)]

bar_run =tqdm.trange(len(lr), colour='#3737ff')
for run in bar_run:
    best_ce_hist = []
    best_acc_hist = []
    bar_run.set_postfix_str('lr = %2.2f' % (lr[run]))
    # plot n stuff
    row, col = run // 3, run % 3
    bar_iteration = tqdm.trange(1, colour='#7f7fff', leave=False)
    for iteration in bar_iteration:
        # data spliting for each iteration
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
        
        # create model, loss function & optimizer
        model = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,3),
            nn.LeakyReLU()
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # traning parameter
        n_epoch = 100
        batch_size = 10
        batch_per_epoch = len(x_train) // batch_size # // is not comment, it is floor division

        # storing result
        best_acc = - np.inf
        best_ce = np.inf
        train_ce_history = []
        train_acc_history = []
        test_ce_history = []
        test_acc_history = []

        
        for epoch in range(n_epoch):
            model.train()
            # bar.set_description(f"Epoch {epoch}")
            for i in range(batch_per_epoch):
                # taking a batch
                start = i * batch_size
                x_batch = x_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                # forward pass
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()


            # update stats & tqdm        
            acc = float((torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean())
            loss = float(loss)
            train_ce_history.append(loss)
            train_acc_history.append(acc)

            # run model through testset (validation)
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test)
                ce = loss_fn(y_pred, y_test).item()
                acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean().item()
            test_ce_history.append(ce)
            test_acc_history.append(acc)
            if ce < best_ce:
                best_ce = ce
                best_weights = copy.deepcopy(model.state_dict())
            if acc >= best_acc: best_acc = acc
            # bar.set_postfix_str(f"validation stats: cross entropy loss = {ce}, acc = {acc}")

        # before go to next iteration of the same lr, record stat
        best_ce_hist.append(best_ce)
        best_acc_hist.append(best_acc)
        bar_iteration.set_postfix_str('best cross entropy loss = %.3f, best acc = %.3f' % (best_ce_hist[iteration], best_acc_hist[iteration]))


    # restore best model
    model.load_state_dict(best_weights)

    sub_1[row][col].plot(train_ce_history, label='train')
    sub_1[row][col].plot(test_ce_history, label='test')
    sub_1[row][col].set_title('Learning Rate = %.6f\nBest loss=%.4f' % (lr[run], max(best_ce_hist)))
    sub_1[row][col].legend()
    sub_2[row][col].plot(train_acc_history, label='train')
    sub_2[row][col].plot(test_acc_history, label='test')
    sub_2[row][col].set_title('Learning Rate = %.6f\nBest accuracy=%.4f' % (lr[run], max(best_acc_hist)))
    sub_2[row][col].legend()
plt.show()

# plots
'''plot, subplot = plt.subplots(1, 2, figsize=(12,5))

subplot[0].plot(train_ce_history, label="train")
subplot[0].plot(test_ce_history, label="test")
subplot[0].set_xlabel("epoch")
subplot[0].set_ylabel("cross entropy")
subplot[0].legend()

subplot[1].plot(train_acc_history, label="train")
subplot[1].plot(test_acc_history, label="test")
subplot[1].set_xlabel("epoch")
subplot[1].set_ylabel("acc")
subplot[1].legend()
plt.show()'''

# save the model
# torch.save(model, "iris-model.pth")

# create new model from saved model
new_model = torch.load("iris-model.pth")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)

# test new model using copied tensor
new_model.eval()
with torch.no_grad():
    y_pred = new_model(x_test)
    acc = float((torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean())
    print(f"acc = {acc}")

    for i in range(10):
        print(x_test[i].tolist(), y_test[i].argmax().item(), y_pred[i].argmax().item())