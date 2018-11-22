
# -*- coding: utf-8 -*-
"""
Implements a mulitlayer LSTM for seq2seq.
"""
#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy.random as rand

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""

from torchtext import data
import torch

vocab_size = 80
batch_size = 20
epochs = 1
path = 'SampleData/train.csv'
glove_dim = 50



TEXT = data.Field(sequential=True)
LABEL = data.Field(sequential=False)
train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
TEXT.build_vocab(train_set, max_size = vocab_size, vectors="glove.6B."+str(glove_dim)+"d")
LABEL.build_vocab(train_set)

#Make glove embedding
vocab = TEXT.vocab
embed = torch.nn.Embedding(len(vocab), glove_dim)

#This is the glove embedding for every word in vocab, we dont need to load it into memory
w = embed.weight.data.copy_(vocab.vectors)


#TODO: Why are the training examples in the columns?
#TODO: How do we enable sort?

dataset_iter = data.Iterator(train_set, batch_size=batch_size, device=0,
        train=True, shuffle=True, repeat=False, sort=False)
    
#Put real training loop here instead
    
for epoch in range(epochs):
    for examples in dataset_iter:
        x = examples.data
        #x=torch.t(x)   
        x1 = examples.data
        y = examples.label
        x = embed(x)        
        print(x)
        print(y)
        break
 

#We now have the data as x and the label as y
#Notice that a column in x corresponds to a training example.
#We can convert from data to string like this:
print()
print('The word corresponding to 28 is:', TEXT.vocab.itos[28]) #What is the word corresponding to 28?

#%% Define global variables
device='cpu'
seqlen=x.size(0)
embedsize=x.size(-1)
batch_size=x.size(1)
layers=2
blocks=40
blocks=seqlen
#%%

class BiEncoderRNN(nn.Module):
    def __init__(self, inputdim, hidden_size):
        super(BiEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size =inputdim, hidden_size=self.hidden_size,
                          bidirectional=True,num_layers=layers)

    def forward(self, x, hidden):
        output=x
        output, hid = self.lstm(output, hidden)

        return output, hid

    def initHidden(self):
        return (torch.zeros(2*layers,batch_size,self.hidden_size 
               , device=device),torch.zeros(2*layers,batch_size,self.hidden_size, device=device))



class BiDecoderRNN(nn.Module):
    def __init__(self, OutDimEnc, hidden_size, output_size):
        super(BiDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size =OutDimEnc, hidden_size=self.hidden_size,
                          bidirectional=True,num_layers=dlayers)
       # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        #output = self.softmax(output[0])
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2*dlayers,batch_size,self.hidden_size 
               , device=device),torch.zeros(2*dlayers,batch_size,self.hidden_size, device=device))

#%%        
enc=BiEncoderRNN(embedsize,blocks)
enc_h=BiEncoderRNN.initHidden(enc)
inp,hid=enc.forward(x,enc_h)

#%% 
dlayers=layers
dblocks=10
decout=vocab_size
decin=blocks


dec=BiDecoderRNN(2*blocks ,dblocks,decout)
dec_h=BiEncoderRNN.initHidden(dec)

