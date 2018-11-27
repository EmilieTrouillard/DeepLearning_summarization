# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""

from torchtext import data
import torch
import numpy as np
from torch import nn

vocab_size = 80
batch_size = 20
epochs = 10
path = '/home/jacob/Downloads/DeepLearning_summarization-master (2)/SampleData/train.csv'
glove_dim = 50
device='cpu'
layers_enc=2 #Num of layers in the encoder
layers_dec=2 #Num of layers in the decoder

h_size_enc = vocab_size #Hidden size of encoder
h_size_dec = vocab_size #Hidden size of decoder

#seqlen=x.size(0)
#embedsize=x.size(-1)



maxlen=966#summary max length
decout=vocab_size

#TODO: How do we enable sort in dataloader?


#%%
"""
Data loader part
"""

TEXT = data.Field(sequential=True)
LABEL = data.Field(sequential=True)
train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
TEXT.build_vocab(train_set, max_size = vocab_size, vectors="glove.6B."+str(glove_dim)+"d")
LABEL.build_vocab(train_set)
vocab = TEXT.vocab
#GloVe mebedding function
embed = torch.nn.Embedding(len(vocab), glove_dim)

#This is the glove embedding for every word in vocab, we dont need to load it into memory
#w = embed.weight.data.copy_(vocab.vectors)

#Data loader iterator
dataset_iter = data.Iterator(train_set, batch_size=batch_size, device=0,
        train=True, shuffle=True, repeat=False, sort=False)

"""
Mask functions
"""

def label_mask(y, y_hat):
    """y_hat is the output tensor from the network
       y is the label tensor (no embedding)
       returns the mask to use for negating the padding"""
    mask = torch.ones(len(y), max(np.shape(y)[1])) 
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

     
#%%
"""
Define the encoder and decoder as well as the forward pass
"""

class BiEncoderRNN(nn.Module):
    def __init__(self, inputdim, hidden_size):
        #Initialize variables in the class
        #TODO: Hidden_size should be vocab size? or 1?        
        super(BiEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=inputdim, hidden_size=self.hidden_size,
                          bidirectional=True,num_layers=layers_enc)

                          
    def forward(self, x, hidden, cell_state):
        output_enc, (hidden_enc, cell_state_enc) = self.lstm(x, (hidden, cell_state))
        return output_enc, (hidden_enc, cell_state_enc)

    def initHidden(self):
        #Initial state for the hidden state and cell state in the encoder
        #Size should be (num_directions*num_layers, batch_size, input_size)
        return (torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device), #h0
                torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device)) #c0



class BiDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        #Initialize variables in the class
        super(BiDecoderRNN, self).__init__()
        self.hidden_size = hidden_size #Should be as long as vocab?
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            bidirectional=False, num_layers=layers_dec)
        
        self.reduce_dim = nn.Linear(2*hidden_size, hidden_size, bias=True)

    def forward(self, attention, input_dec, hidden_enc, cell_state_enc):
        #During training input_dec should be the label (sequence), during test it should the previous output (one word)

        #We reduce the forwards and backwards direction hidden states into
        #one, as our decoder is unidirectional.
        old_enc = torch.cat((hidden_enc[0:2],hidden_enc[2:]),dim=-1)
        old_cell = torch.cat((cell_state_enc[0:2],cell_state_enc[2:]),dim=-1)
        new_enc = self.reduce_dim(old_enc)
        new_cell = self.reduce_dim(old_cell)

        output_dec, (hidden_dec, cell_state_dec) = self.lstm(input_dec, (new_enc, new_cell))
        return output_dec, (hidden_dec, cell_state_dec)

def forward_pass(encoder, decoder, x, label):
    """
    Executes a forward pass through the whole model.
    :param encoder:
    :param decoder:
    :param x: input to the encoder, shape [batch, seq_in_len]
    :param t: target output predictions for decoder, shape [batch, seq_t_len]
    :param criterion: loss function
    :param max_t_len: maximum target length

    :return: output (after log-softmax), loss, accuracy (per-symbol)
    """

    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden()
    
    #Run encoder and get last hidden state (and output).

    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, hidden_enc, cell_state_enc)
    
    attention = 0 #TODO: Implement attention
    
    output, (hidden_dec, cell_state_dec) = decoder.forward(attention, label, hidden_enc, cell_state_enc)

#    loss = criterion(output, t)
#    output=torch.argmax(output, dim=-1)
    #accuracy = (output == t).type(torch.FloatTensor).mean()
    return output

        
#%%
#Put real training loop here instead
    
#for epoch in range(epochs):
for examples in dataset_iter:
    x = examples.data
    y = examples.label
    y = torch.reshape(y,(np.shape(y)[0],np.shape(y)[1],1))
    y = y.float()
    x = embed(x)        
    break
    
encoder = BiEncoderRNN(glove_dim, 10)
decoeder = BiDecoderRNN(1, 10)

forward_pass(encoder, decoeder, x, y)
#%%        
#encoder = BiEncoderRNN(glove_dim, h_size_enc)
#enc_h = BiEncoderRNN.initHidden(encoder)
#
#inp,hid=encoder.forward(x,enc_h)
#
##%% 
#
#dec=BiDecoderRNN(blocks ,dblocks,decout)
#dec_h=BiEncoderRNN.initHidden(dec)
#outp,loss,dechid=dec.forward(hid[0],dec_h,y)
##%%
#
#enc=BiEncoderRNN(embedsize,blocks)
#criterion=nn.CrossEntropyLoss()

#%%
#i=forward_pass(enc,dec,x=x,label=y)

#%%
def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len):
    
    encoder.train()
    decoder.train()
    for batch_idx, (x, t, t_in) in enumerate(zip(inputs, targets, targets_in)):
        

        
        
        if batch_idx % 200 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tTraining loss: {:.4f} \tTraining accuracy: {:.1f}%'.format(
            epoch, batch_idx * len(x), TRAINING_SIZE, 
            100. * batch_idx * len(x) / TRAINING_SIZE, loss.item(), 100. * accuracy.item()))

#%% Training op

#enc = BiEncoderRNN(glove_dim, )
