import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet
with open('intents.json', 'r') as f:
    intents = json.load(f)

# applying our NLP Preprocessing Pipeling to intents: tokenize -> lower + stem -> bag of words
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
tags = sorted(set(tags))
ignore_items = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_items]
all_words = sorted(set(all_words))

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

# training data
x_train = np.array(x_train)
y_train = np.array(y_train)

# create pytorch dataset from the training data
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
# Hyperparameters
batch_size = 9 #during training, model processes 8 samples at a time
hidden_size = 9
output_size = len(tags) # number of different tags we have
input_size = len(x_train[0]) # number of bag of words we created
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        # empty gradients first:
        optimizer.zero_grad()
        loss.backward() # back propagation
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags
}

FILE = "data.pth"
torch.save(data, FILE) # serialize and save

