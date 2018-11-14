# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""

from torchtext import data
from torchtext import vocab
import torch

vocab_size = 80
batch_size = 30
epochs = 1
path = '/home/jacob/Downloads/DeepLearning_summarization-master (2)/SampleData/train.csv'

TEXT = data.Field(sequential=True)
LABEL = data.Field(sequential=False)
train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
TEXT.build_vocab(train_set, max_size = vocab_size, vectors="glove.6B.50d")
LABEL.build_vocab(train_set)

#TODO: How do we see the GloVe vector representation?
#TODO: Why are the training examples in the columns?
#TODO: How do we enable sort?

dataset_iter = data.Iterator(train_set, batch_size=batch_size, device=0,
        train=True, shuffle=True, repeat=False, sort=False)
    
#Put real training loop here instead
    
for epoch in range(epochs):
    for examples in dataset_iter:
        x = examples.data
        y = examples.label
        print(x)
        print(y)
        break
    
#We now have the data as x and the label as y
#Notice that a column in x corresponds to a training example.
#We can convert from data to string like this:
print()
print('The word corresponding to 28 is:', TEXT.vocab.itos[28]) #What is the word corresponding to 28?
