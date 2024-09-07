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
print(xy)
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

    def __getItem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
# Hyperparameters
batch_size = 8 # during training, model processes 8 samples at a time
hidden_size = 8
output_size = len(tags) # number of different tags we have
input_size = len(x_train[0]) # number of bag of words we created
print(input_size, len(all_words), output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = NeuralNet(input_size, hidden_size, output_size)
