#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# - Tutorial source: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# - Demonstrate how to use the text classification datasets in `torchtext` library

# # Let's Begin!

# ## Load data with ngrams

import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Define the Model

import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


# ## Initiate an instance

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)


# ## Functions used to generate batch

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


# ## Define functions to train the model and evaluate results.

from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


# ## Split the dataset and run the model

import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ =     random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


# ## Evaluation with Test dataset

print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


# ## Test on a random news

import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was     enduring the season’s worst weather conditions on Sunday at The     Open on his way to a closing 75 at Royal Portrush, which     considering the wind and the rain was a respectable showing.     Thursday’s first round at the WGC-FedEx St. Jude Invitational     was another story. With temperatures in the mid-80s and hardly any     wind, the Spaniard was 13 strokes better in a flawless round.     Thanks to his best putting performance on the PGA Tour, Rahm     finished with an 8-under 62 for a three-stroke lead, which     was even more impressive considering he’d never played the     front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])


ex_text_str = """
    The FAA seems satisfied with its investigations into Elon Musk’s last two SpaceX Starship 
    tests, each of which ended in an explosive crash, and the conclusion of those investigations 
    should clear the way for a new SN10 flight in the very near future. 
    In fact, Musk just tweeted there’s a "good chance of flying this week!"
    Late last month, we broke the news that SpaceX had violated its launch license with its Starship SN8 launch in December, but an FAA spokesperson now says that matter has already been settled, according to CNN’s Jackie Wattles.
"""

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])


ex_text_str = """
    After a seven-month-long journey, NASA’s Perseverance Rover successfully touched down 
    on the Red Planet on February 18, 2021. 
    
    Mission controllers at NASA’s Jet Propulsion Laboratory in Southern California 
    celebrate landing NASA’s fifth — and most ambitious — rover on Mars.

    A key objective for Perseverance’s mission on Mars is astrobiology, including 
    the search for signs of ancient microbial life. The rover will characterize 
    the planet’s geology and past climate, pave the way for human exploration of the Red Planet, 
    and be the first mission to collect and cache Martian rock and regolith.

    Also flying with Perseverance is NASA’s Ingenuity helicopter, which will attempt to show 
    controlled, powered flight is possible in the very thin Martian atmosphere.
"""

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])


ex_text_str = """
    Hundreds of thousands of protesters have taken to the streets of Myanmar in one of the 
    largest demonstrations yet against the country's military coup.

    Businesses closed as employees joined a general strike, despite a military statement 
    that said protesters were risking their lives by turning out.

    Police dispersed crowds in the capital, Nay Pyi Taw, and a water cannon truck was 
    seen moving into position.

    Myanmar has seen weeks of protest following the coup on 1 February. 
    Military leaders overthrew Aung San Suu Kyi's elected government and have placed her 
    under house arrest, charging her with possessing illegal walkie-talkies and violating 
    the country's Natural Disaster Law.
"""

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])




