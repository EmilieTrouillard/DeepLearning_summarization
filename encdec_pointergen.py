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
import copy
import socket

vocab_size = 75 #Size of the vocab
batch_size = 20 #Batch size
epochs = 50 #How many epochs we train
attention_features = 1000 #The number of features we calculate in the attention (Row amount of Wh, abigail eq 1)
vocab_features = 1000 #The number of features we calculate when we calculate the vocab (abigail eq 4)
LEARNING_RATE = 0.001
LAMBDA_COVERAGE = 2
layers_enc=2 #Num of layers in the encoder
layers_dec=2 #Num of layers in the decoder

hidden_size = 100 #Hiddensize dimension (double the size for encoder, because bidirectional)


save_model = True
load_model = False


if socket.gethostname() == 'jacob':
    path = '/home/jacob/Desktop/DeepLearning_summarization/Data/train_medium_unique.csv'
    path_val = '/home/jacob/Desktop/DeepLearning_summarization/Data/validation_medium_unique.csv'
    PATH = '/home/jacob/Desktop/DeepLearning_summarization/'
else:
    path = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/train_short.csv'
    path_val = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/validation_short.csv'
    PATH = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/saved_network'
glove_dim = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#TODO: How do we enable sort in dataloader?


#%%
"""
Data loader part
"""

TEXT = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
LABEL = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
validation_set = data.TabularDataset(path_val, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 

TEXT.build_vocab(train_set, vectors="glove.6B."+str(glove_dim)+"d")
LABEL.vocab = TEXT.vocab
vocab = copy.deepcopy(TEXT.vocab)

#GloVe embedding function
embed = torch.nn.Embedding(len(vocab), glove_dim)

#This is the glove embedding for every word in vocab, we dont need to load it into memory
#w = embed.weight.data.copy_(vocab.vectors)

#Data loader iterator
dataset_iter = data.Iterator(train_set, batch_size=batch_size, device=0,
        train=True, shuffle=True, repeat=False, sort=False)

dataset_iter_val = data.Iterator(validation_set, batch_size=1, device=0,
        train=True, shuffle=True, repeat=False, sort=False)

"""
Mask functions
"""

def label_mask(y):
    """y is the label tensor (no glove embedding)
       returns the mask to use for negating the padding"""
    mask = (y[1:] != 1)
    return mask.float().cuda()

def attention_mask(x):
    """x is the training data tensor (no embedding)
       returns the mask to use for negating the padding effect on the attention
       add this mask before taking the softmax!"""
    mask = torch.zeros(len(x), len(x[0]), 1, device=device)
    for i in range(len(x)):
        try:
            index = np.where(x[i]==1)[0][0]
            mask[i][index:] = -np.inf
        except:
            pass
    return mask


def display(x, vocab_ext):
    return ' '.join([vocab_ext.itos[i] for i in x if vocab_ext.itos[i] != '<pad>'])
     
#%%
"""
Define the encoder and decoder as well as the forward pass
"""

class BiEncoderRNN(nn.Module):
    def __init__(self, inputdim, hidden_size):
        #Initialize variables in the class
        super(BiEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=inputdim, hidden_size=self.hidden_size,
                          bidirectional=True,num_layers=layers_enc)

                          
    def forward(self, x, hidden, cell_state):
        output_enc, (hidden_enc, cell_state_enc) = self.lstm(x, (hidden, cell_state))
        return output_enc, (hidden_enc, cell_state_enc)

    def initHidden(self, train=True):
        #Initial state for the hidden state and cell state in the encoder
        #Size should be (num_directions*num_layers, batch_size, input_size)
        if train:
            return (torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device), #h0
                    torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device)) #c0
        else:
            return (torch.zeros(2*layers_enc, 1, self.hidden_size, device=device), #h0
                    torch.zeros(2*layers_enc, 1, self.hidden_size, device=device)) #c0



class BiDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        #Initialize variables in the class
        super(BiDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            bidirectional=False, num_layers=layers_dec)
        
        self.reduce_dim = nn.Linear(2*hidden_size, hidden_size, bias=True)
        self.linearOut = nn.Linear(hidden_size, vocab_size) #Not used when we apply attention
        self.attention = nn.Linear(2*self.hidden_size + self.hidden_size + 1, attention_features, bias=True)
        self.tanh = nn.Tanh()
        self.attn_out = nn.Linear(attention_features, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)
        self.softmaxvocab = nn.Softmax(dim=2)
        self.linearVocab1 = nn.Linear(2*hidden_size+hidden_size,vocab_features, bias=True) 
        self.linearVocab2 = nn.Linear(vocab_features, vocab_size, bias=True) 
        self.sigmoid = nn.Sigmoid()
        self.pgen_linear = nn.Linear(2*hidden_size + hidden_size + 1, 1, bias=True)

    def forward(self, output_enc, real_index, coverage, input_dec, hidden_enc, cell_state_enc, att_mask, vocab_ext, first_word=True):
        #During training input_dec should be the label (sequence), during test it should the previous output (one word)
        #We reduce the forwards and backwards direction hidden states into
        #one, as our decoder is unidirectional.
        if first_word:
            old_enc = torch.cat((hidden_enc[0:2],hidden_enc[2:]),dim=-1)
            old_cell = torch.cat((cell_state_enc[0:2],cell_state_enc[2:]),dim=-1)
            new_enc = self.reduce_dim(old_enc)
            new_cell = self.reduce_dim(old_cell)
            coverage = torch.zeros(output_enc.size()[0], output_enc.size()[1], 1, device=device)
        else:
            new_enc = hidden_enc
            new_cell = cell_state_enc
        output_dec, (hidden_dec, cell_state_dec) = self.lstm(input_dec, (new_enc, new_cell))
        pvocab = torch.zeros((len(output_dec), batch_size, max(vocab_size,int(torch.max(real_index)+1)))).cuda() 
        coverage_loss = torch.zeros((len(output_dec), batch_size, 1)).cuda()
        
        #pgen = torch.zeros(len(output_dec),batch_size,1).cuda()
        pgen = torch.zeros(batch_size, 1).cuda()        
        #Attention
        #We loop over t in the equation
        for t in range(len(output_dec)):
            #Expand negates a loop over i
            #Output_dec contains all decoder states at times t=0, ... , t=N
            input_attention = torch.cat((output_enc, output_dec[t:t+1].expand(len(output_enc), batch_size, self.hidden_size), coverage), dim=2)
            e = self.attention(input_attention)
            e = self.tanh(e)
            e = self.attn_out(e)
            e = e + att_mask
            attention = self.softmax(e)
      
            coverage_loss[t] = torch.sum(torch.min(attention, coverage), dim=0)
            coverage = coverage + attention
            
            #Calculate context vector sum(a_i^t * h_i)
            #This becomes a weighted sum of hidden states, hidden states created
            #after inputing a word with high attention has more weight
            context = torch.sum(attention * output_enc, dim=0).reshape((1,batch_size, 2*self.hidden_size))
                     
            
            #Calculate pointer generation probability
            in_pgen = torch.cat((context, output_dec[t:t+1], input_dec[t:t+1]),dim=2)
            
            pgen = self.pgen_linear(in_pgen)
            pgen = self.sigmoid(pgen)
            #pgen[t] = self.pgen_linear(in_pgen)
            #pgen[t] = self.sigmoid(pgen[t])
            
            
            #Calculate Pvocab (no softmax in training loop as it is included in cross entropy loss)
            p_vocab = torch.cat((output_dec[t:t+1],context),2)
            p_vocab = self.linearVocab1(p_vocab)
            p_vocab = self.linearVocab2(p_vocab)
            p_vocab = self.softmaxvocab(p_vocab)

            #Multiply pvocab with pointer generation probability
            p_vocab = p_vocab * pgen
            #p_vocab is 1 x batch_size x vocab size
            if torch.max(real_index) >= vocab_size:
                p_vocab = torch.cat((p_vocab, torch.zeros(1,batch_size,int(torch.max(real_index)+1-vocab_size)).cuda()), dim=2)
            for i in range(batch_size):
                for j in range(attention.size()[0]):
                    if real_index[j,i] == 1:
                        break
                    else:
                        index = int(real_index[j,i])
                        p_vocab[0,i,index] = p_vocab[0,i,index] + attention[j,i,0]*(1-pgen[0,i,0])
            pvocab[t] = p_vocab
        
        return pvocab, coverage_loss, coverage, (hidden_dec, cell_state_dec)
 #reduction='none' because we want one loss per element and then apply the mask
def forward_pass(encoder, decoder, real_index, x, label, criterion, att_mask, vocab_ext):
    """
    Executes a forward pass through the whole model.
    :param encoder:
    :param decoder:
    :param real_text: the input in real text
    :param x: input to the encoder, shape [batch, seq_in_len]
    :param t: target output predictions for decoder, shape [batch, seq_t_len]
    :param criterion: loss function
    :param max_t_len: maximum target length

    :return: output (after log-softmax), loss, accuracy (per-symbol)
    """

    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden(train=True)
    
    #Run encoder and get last hidden state (and output).

    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, hidden_enc, cell_state_enc)
    
    output, cov_loss, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label[:-1], hidden_enc, cell_state_enc, att_mask, vocab_ext)
    
    label_hat = torch.argmax(output, -1)

    output = output.permute([0,2,1]) #N,C,d format where C number of classes for the Cross Entropy    
    loss = criterion(output.unsqueeze(3), label[1:].long())

  
    combined_loss = loss + LAMBDA_COVERAGE * cov_loss

    mask = label_mask(label)
    loss_mask = torch.sum(combined_loss * mask, dim=0) 
    loss_batch = loss_mask / torch.sum(mask, dim=0)
    total_loss = torch.mean(loss_batch)
    
    loss_maskCE = torch.sum(loss * mask, dim=0) 
    loss_batchCE = loss_maskCE / torch.sum(mask, dim=0)
    total_lossCE = torch.mean(loss_batchCE)
    
    return output, total_lossCE, total_loss, label_hat

        
def forward_pass_val(encoder, decoder, real_index, x, att_mask, vocab_ext):
    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden(train=False)
    
    #Run encoder and get last hidden state (and output).

    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, hidden_enc, cell_state_enc)
    EOS = False
    label_hat = torch.Tensor(1,1,1).cuda()
    label_hat[:] = 2
    out = []
    output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label_hat, hidden_enc, cell_state_enc, att_mask, vocab_ext, first_word=True)
    index = torch.argmax(output, -1)
    label_hat[:] = index
    if label_hat == 3:
        EOS = True
    out.append(index)
    while not EOS and len(out) < len(x):
        output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, coverage, label_hat, hidden_dec, cell_state_dec, att_mask, vocab_ext, first_word=False)
        index = torch.argmax(output, -1)
        label_hat[:] = index
        if label_hat == 3:
            EOS = True
        out.append(index)
    return torch.stack(out)


#%%
def train(encoder, decoder, data, criterion, enc_optimizer, dec_optimizer, epoch):
    encoder.train()
    decoder.train()
    i = 0
    for batchData in data:
        vocab_ext = copy.deepcopy(vocab)
        real_index = batchData.data
        att_mask = attention_mask(real_index)
        y1 = batchData.label
        y = torch.reshape(y1,(np.shape(y1)[0], np.shape(y1)[1], 1))
        y = y.float().cuda()
        x = embed(real_index).cuda()
        i += 1

        out, crossEnt_loss, loss, label_hat = forward_pass(encoder, decoder, real_index, x, y, criterion, att_mask, vocab_ext)
        
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        if i % 1 == 0:
            print('Epoch {} [Batch {}]\tTraining loss: {:.4f} \tCoverage-CE ratio: :{:.4f}'.format(
                epoch, i, loss.item(), (loss.item() - crossEnt_loss)/crossEnt_loss))
        if i % 20 == 0:
            #print('input')
            #print(display(real_index, vocab_ext))
            print('output')
            try:
                ind = np.where(y1[1:,0] == 1)[0][0]
                print(display(label_hat[:ind,0], vocab_ext))
            except:
                print(display(label_hat[:,0], vocab_ext))
            print('real output')
            print(display(y1[1:,0], vocab_ext))
            

def validation(encoder, decoder, data):
    acc = []
    for batchData in data:
        i = 0
        vocab_ext = copy.deepcopy(vocab)
        real_index = batchData.data
        att_mask = torch.zeros(real_index.size()[0], real_index.size()[1], 1, device=device)
        y = batchData.label
        y = torch.reshape(y,(np.shape(y)[0], np.shape(y)[1], 1))
        y = y.long().cuda()
        x = embed(real_index).cuda()
        test_output = forward_pass_val(encoder, decoder, real_index, x, att_mask, vocab_ext)
        test_output = test_output.cuda()
        i += 1
        if len(test_output) == len(y[1:]):
            if len(test_output) == sum(test_output == y[1:]):
                acc.append[1]
            else:
                acc.append[0]
    return acc

#%% Training op
if load_model:
    encoder = torch.load(PATH + '_encoder_pgen')
    decoder = torch.load(PATH + '_decoder_pgen')
else:
    encoder = BiEncoderRNN(glove_dim, hidden_size).to(device)
    decoder = BiDecoderRNN(1, hidden_size).to(device)
enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction='none')

#%%

try:
    for epoch in range(1, epochs + 1):
        batch_size=batch_size
        train(encoder, decoder, dataset_iter, criterion, enc_optimizer, dec_optimizer, epoch)
    batch_size=1
    acc = validation(encoder, decoder, dataset_iter_val)
    if save_model:
        torch.save(encoder, PATH + '_encoder_pgen')
        torch.save(decoder, PATH + '_decoder_pgen')
except KeyboardInterrupt:
    if save_model:
        torch.save(encoder, PATH + '_encoder_pgen')
        torch.save(decoder, PATH + '_decoder_pgen')