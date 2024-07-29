from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import pandas as pd

# get and write data
data = fetch_california_housing()
print(data.feature_names)
x,y = data.data, data.target

# create a model
model = nn.Sequential(
    nn.Linear(8,32),
    nn.ReLU(),
    
    nn.Linear(32,16),
    nn.ReLU(),
    
    nn.Linear(16,8),
    nn.ReLU(),
    
    nn.Linear(8,4),
    nn.ReLU(),
    
    nn.Linear(4,1)
)

# loss function and optimizer
loss_fn = nn.MSELoss() #mean sq error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# split training and test from dataset
x_train_raw, x_test_raw, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True)

# standardizing data
scaler = StandardScaler()
scaler.fit(x_train_raw)
x_train = scaler.transform(x_train_raw)
x_test = scaler.transform(x_test_raw)

# convert to 2D pytorch tensor
x_train = torch.tensor(x_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32).reshape(-1,1)
x_test = torch.tensor(x_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32).reshape(-1,1)

# training parameter
n_epoch = 50
batch_size = 20
batch_start = torch.arange(0, len(x_train), batch_size)

# hold the best model
best_mse = np.inf
best_weight = None
mse_hist = [[], []]

#training loop
with tqdm.tqdm(range(n_epoch), unit = 'epoch') as bar:
    for epoch in bar:
        
        bar.set_description(f"Epoch {epoch}")
        model.train()
        for start in batch_start:

            # take a batch
            x_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            # forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weight
            optimizer.step()

        mse_hist[0].append(float(loss))

        # evaluate accuracy at end of each epoch
        model.eval()
        torch.no_grad()
        y_pred = model(x_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        mse_hist[1].append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weight = copy.deepcopy(model.state_dict())
        bar.set_postfix(train_mse = float(loss), test_mse = float(mse), best_mse = best_mse)
        torch.enable_grad()

model.eval()
with torch.no_grad():
    pred = model(x_test).numpy()


feature = pd.DataFrame(data=x_test.numpy(), columns=data.feature_names)
label = pd.DataFrame(data=y_test.numpy(), columns=['HousePrice'])
predict = pd.DataFrame(data=pred, columns=['Prediction'])
result = pd.concat([feature, label, predict], axis=1)
print(result)
result.to_csv("result.csv")

# restore model and return best accuracy
model.load_state_dict(best_weight)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

# test model real time
model.eval()
with torch.no_grad():
    # test out inference with 5 samples
    for i in range(5):
        x_sample = x_test_raw[i:i+1]
        x_sample = scaler.transform(x_sample)
        x_sample = torch.tensor(x_sample, dtype=torch.float32)
        y_pred = model(x_sample)
        print(f"{x_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

# plot
name = ["Train MSE", "Test MSE"]
plt.plot(mse_hist[0], label=name[0])
plt.plot(mse_hist[1], label=name[1])
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Mean Square Error (MSE) loss history of model")
plt.legend()
plt.show()