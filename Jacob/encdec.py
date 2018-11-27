# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""

from torchtext import data
import torch
import numpy as np
from torch import nn
import torch.optim as optim

vocab_size = 80
batch_size = 20
epochs = 10
#path = '/home/jacob/Downloads/DeepLearning_summarization-master (2)/SampleData/train.csv'
path = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/train.csv'
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

TEXT = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
LABEL = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
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
    batch_size = np.shape(y)[1]
    max_len = max(len(y), len(y_hat))
    mask = torch.ones(max_len, batch_size) 
    for i in range(batch_size):
        try:
            y_hat_index =  np.where(y_hat[:,i]==1)[0][0]
            y_index = np.where(y[:,i]==1)[0][0]
            index = max(y_hat_index, y_index)
            mask[index:, i] = 0
        except:
            pass
    return mask.float()

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


def display(x):
    return ' '.join([TEXT.vocab.itos[i] for i in x[:,0] if TEXT.vocab.itos[i] != '<pad>'])
     
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
        self.linearOut = nn.Linear(hidden_size, vocab_size) #TODO: bias?


    def forward(self, attention, input_dec, hidden_enc, cell_state_enc):
        #During training input_dec should be the label (sequence), during test it should the previous output (one word)

        #We reduce the forwards and backwards direction hidden states into
        #one, as our decoder is unidirectional.
        old_enc = torch.cat((hidden_enc[0:2],hidden_enc[2:]),dim=-1)
        old_cell = torch.cat((cell_state_enc[0:2],cell_state_enc[2:]),dim=-1)
        new_enc = self.reduce_dim(old_enc)
        new_cell = self.reduce_dim(old_cell)

        output_dec, (hidden_dec, cell_state_dec) = self.lstm(input_dec, (new_enc, new_cell))
        output_dec = self.linearOut(output_dec)
        return output_dec, (hidden_dec, cell_state_dec)

 #reduction='none' because we want one loss per element and then apply the mask
def forward_pass(encoder, decoder, x, label, criterion):
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
    
    out = output.permute([0,2,1]) #N,C,d format where C number of classes for the Cross Entropy

    label_hat = torch.argmax(output, -1)
    loss = criterion(out, torch.squeeze(label).long())
    mask = label_mask(label, label_hat)
    loss_mask = torch.sum(loss * mask, dim=0) 
    loss_batch = loss_mask / torch.sum(mask, dim=0)
    mean_loss = torch.mean(loss_batch)
    #accuracy = (output == t).type(torch.FloatTensor).mean()
    return output, mean_loss, label_hat

        
#%%
#Put real training loop here instead
    
#for epoch in range(epochs):
for examples in dataset_iter:
    x1 = examples.data
    y = examples.label
    y = torch.reshape(y,(np.shape(y)[0],np.shape(y)[1],1))
    y = y.float()
    x = embed(x1)        
    break
    
encoder = BiEncoderRNN(glove_dim, 10)
decoder = BiDecoderRNN(1, 10)

#out, loss, label_hat = forward_pass(encoder, decoder, x, y, criterion)
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
def train(encoder, decoder, data, criterion, enc_optimizer, dec_optimizer, epoch):
    encoder.train()
    decoder.train()
    i = 0
    for batchData in data:
        i += 1
        x1 = batchData.data
        y = batchData.label
        y = torch.reshape(y,(np.shape(y)[0], np.shape(y)[1], 1))
        y = y.float()
        x = embed(x1)
        out, loss, label_hat = forward_pass(encoder, decoder, x, y, criterion)
        
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        
        if i % 5 == 0:
            print('Epoch {} [Batch {}]\tTraining loss: {:.4f}'.format(
                epoch, i, loss.item()))
            print('input')
            print(display(x1))
            print('output')
            print(display(label_hat))

#%% Training op
LEARNING_RATE = 0.003
enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction='none')

#%%
EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    train(encoder, decoder, dataset_iter, criterion, enc_optimizer, dec_optimizer, epoch)