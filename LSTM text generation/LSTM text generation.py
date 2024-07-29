import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np

# i need to use my computer bruh
# torch.set_num_threads(5)

## prepare data
# load the file in ascii
filename = "Alice in Wanderland.txt"
raw_text = open(filename, "r", encoding='utf-8').read()
raw_text = raw_text.lower() # conver to lower case


# create mapping for unique character
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

# summarize the loaded data
n_char = len(raw_text)
n_vocab = len(chars)
print(f"Total Characters: {n_char}, Total Vocab: {n_vocab}")

# prepare dataset of input to output pairs encoded as integers
seq_len = 100
x_data, y_data = [], []
for i in range(0, n_char - seq_len, 1):
    seq_in = raw_text[i : i + seq_len]
    seq_out = raw_text[i + seq_len]
    x_data.append([char_to_int[char] for char in seq_in])
    y_data.append(char_to_int[seq_out])
n_pattern = len(x_data)
print(f"Total patterns: {n_pattern}")

# convert data to tensor
x = torch.tensor(x_data, dtype=torch.float32).reshape(n_pattern, seq_len, 1) # torch.size[sample, time step, feature]
x = x / float(n_vocab) # normalize
y = torch.tensor(y_data)
print(x.size(), y.size())

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :] # take only last output
        x = self.linear(self.dropout(x))
        return x
    
## parameters
n_epoch = 40
batch_size = 128
model = LSTM()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(x, y), shuffle=True, batch_size=batch_size)
best_model = None
best_loss = np.inf

## training
'''import tqdm
prog_epoch = tqdm.tqdm(range(n_epoch), unit='batch', mininterval=1, position=0)
for epoch in prog_epoch:
    prog_train = tqdm.tqdm(loader, unit='epoch', mininterval=1, colour='#00ff00', desc='Training', position=1)
    model.train()
    for x_batch, y_batch in prog_train:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        loss = 0
        prog_test = tqdm.tqdm(loader, unit='epoch', mininterval=1, colour='#7f7f00', desc='Testing', position=1)
        for x_batch, y_batch in prog_test:
            y_pred = model(x_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
    torch.save([best_model, char_to_int], "single-char-2-layer.pth")

    prog_epoch.set_postfix_str(f"loss = {loss.item()}, best_loss = {best_loss.item()} at epoch {epoch}")'''


## model in action
model = LSTM()
loaded_state_dict, char_to_int = torch.load('single-char-2-layer.pth')
model.load_state_dict(loaded_state_dict)
int_to_char = dict((i, c) for c, i in char_to_int.items())
start = np.random.randint(0, len(raw_text) - seq_len)
prompt = raw_text[start : start + seq_len]
prompt = "                  “what is the secret of the universe?” asked alice\n“the secret of the universe is”"
prompt_int = [char_to_int[c] for c in prompt]

model.eval()
print("Prompt:\n%s\n\nGenerated:\n" % prompt)
with torch.no_grad():
    for i in range(2000):
        # format input array of int into Pytorch tensor
        x = np.reshape(prompt_int, (1, len(prompt_int), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)

        # generate logit as output
        prediction = model(x)  # torch.size([1, num_vocab])

        # convert logits as output from the model
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")

        # append the new charater into prompt for next iteration
        prompt_int.append(index)
        prompt_int = prompt_int[1:] # cut the first charater

print("\nFinished Generating")
