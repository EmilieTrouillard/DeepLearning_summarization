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
#%% Define global variables
device='cpu'
inputdim=10
batch_size=5
#%%Get data
#Dont understand the data-generator, creating random data
randdat=rand.randint(size=(inputdim,batch_size)) #
dat=torch.from_numpy(randdat)

#%% Define network

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, self.hidden_size)
        rnn = nn.GRU
        self.rnn = rnn(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs, hidden):

        inputs = inputs.long()

        embedded = self.embedding(inputs)

        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        init = torch.zeros(2, inputdim, batch_size, device=device)#2 cause bidiretional
        return init
encoder=EncoderRNN(inputdim,batch_size)
#%%
enc_h = encoder.init_hidden(batch_size)
enc_out, enc_h = encoder(dat, enc_h)

#%%

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        rnn = nn.GRU
        self.rnn = rnn(self.hidden_size, self.hidden_size, batch_first=True,bidirectional=True)

    def forward(self, inputs, hidden, output_len):
        inputs=inputs.long()
        dec_input = inputs[:, 0].unsqueeze(1)
        output = []
        for i in range(output_len):
            out, hidden = self.rnn(self.embedding(dec_input), hidden)
            out = self.out(out) 
            out = F.log_softmax(out, -1)
            output.append(out.squeeze(1))
            out_symbol = torch.argmax(out, dim=2)   # shape [batch, 1]
            dec_input = out_symbol   # feed the decoded symbol back into the recurrent unit at next step

        output = torch.stack(output).permute(1, 0, 2)  # [batch_size x seq_len x output_size]

        return output
DecInSize=dec_h.size(1)#hidden states previous
max_out=10
decoder=DecoderRNN(inputdim,max_out)
#%%
dec_h = enc_h

dec_input = enc_out

out = decoder(dec_input, dec_h, max_out)


#%%
def forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, teacher_forcing):
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
    # Run encoder and get last hidden state (and output)
    batch_size = x.size(0)
    enc_h = encoder.init_hidden(batch_size)
    enc_out, enc_h = encoder(x, enc_h)

    dec_h = enc_h  # Init hidden state of decoder as hidden state of encoder
    dec_input = t_in
    out = decoder(dec_input, dec_h, max_t_len, teacher_forcing)
    out = out.permute(0, 2, 1)
    # Shape: [batch_size x num_classes x out_sequence_len], with second dim containing log probabilities

    loss = criterion(out, t)
    pred = get_pred(log_probs=out)
    accuracy = (pred == t).type(torch.FloatTensor).mean()
    return out, loss, accuracy


     

#%%
def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len):
    encoder.train()
    decoder.train()
    for batch_idx, (x, t, t_in) in enumerate(zip(inputs, targets, targets_in)):
        
        
        # INSERT YOUR CODE HERE
        
        
        if batch_idx % 200 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tTraining loss: {:.4f} \tTraining accuracy: {:.1f}%'.format(
                epoch, batch_idx * len(x), TRAINING_SIZE,
                100. * batch_idx * len(x) / TRAINING_SIZE, loss.item(),
                100. * accuracy.item()))


#%% 
