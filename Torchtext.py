# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""

from torchtext import data
import torch
import numpy as np

vocab_size = 800
batch_size = 20
epochs = 1
path = '/home/jacob/Downloads/DeepLearning_summarization-master (2)/SampleData/train.csv'
glove_dim = 50



TEXT = data.Field(sequential=True)
LABEL = data.Field(sequential=True)
train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
TEXT.build_vocab(train_set, max_size = vocab_size, vectors="glove.6B."+str(glove_dim)+"d")
LABEL.build_vocab(train_set)

#Make glove embedding
vocab = TEXT.vocab
embed = torch.nn.Embedding(len(vocab), glove_dim)

#This is the glove embedding for every word in vocab, we dont need to load it into memory
#w = embed.weight.data.copy_(vocab.vectors)


def label_mask(y, y_hat):
    """y_hat is the output tensor from the network
       y is the label tensor (no embedding)
       returns the mask to use for negating the padding"""
    mask = torch.ones(len(y), max(np.shape(y)[1], np.shape(y_hat)[1])) 
    for i in range(len(y)):
        y_hat_index = len(y_hat) - 1
        try:
            y_index = np.where(y[i]==1)[0][0]
            index = max(y_hat_index, y_index)
            mask[i,index:] = 0
        except:
            pass
    return mask

def attention_mask(x):
    """x is the training data tensor (no embedding)
       returns the mask to use for negating the padding effect on the attention
       add this mask before taking the softmax!"""
    mask = torch.zeros(len(x), len(x[0]))
    for i in range(len(x)):
        try:
            index = np.where(x[i]==1)[0][0]
            mask[i][index:] = -np.inf
        except:
            pass
    return mask


#TODO: Why are the training examples in the columns?
#TODO: How do we enable sort?

dataset_iter = data.Iterator(train_set, batch_size=batch_size, device=0,
        train=True, shuffle=True, repeat=False, sort=False)
    
#Put real training loop here instead
    
for epoch in range(epochs):
    for examples in dataset_iter:
        x = examples.data
        y = torch.t(examples.label)
        x = embed(torch.t(x))        
        print(x)
        print(y[0])
        break
 
   
#We now have the data as x and the label as y
#Notice that a column in x corresponds to a training example.
#We can convert from data to string like this:
print()
print('The word corresponding to 28 is:', TEXT.vocab.itos[28]) #What is the word corresponding to 28?
