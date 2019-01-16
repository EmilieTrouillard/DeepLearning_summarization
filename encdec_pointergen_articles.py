#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:27:39 2018

@author: jacob
"""
dataset_type = 'articles'
#dataset_type = 'dummy'

from torchtext import data
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import copy
import socket
from torchtext.data import ReversibleField, BucketIterator
import os
import time
import matplotlib.pyplot as plt

if dataset_type == 'articles':
    from cnn_dm_torchtext_master.summarization import CNN, DailyMail

vocab_size = 50000 if dataset_type == 'articles' else 40 #Size of the vocab
BATCH_SIZE = 12 if dataset_type == 'articles' else 23  #Batch size
epochs = 50 #How many epochs we train
attention_features = 100 #The number of features we calculate in the attention (Row amount of Wh, abigail eq 1)
vocab_features = 100 #The number of features we calculate when we calculate the vocab (abigail eq 4)
LEARNING_RATE = 0.0001
LAMBDA_COVERAGE = 1
layers_enc=2 #Num of layers in the encoder
layers_dec=2 #Num of layers in the decoder
MAX_LENGTH = 100
hidden_size = 256 #Hiddensize dimension (double the size for encoder, because bidirectional)
TRUNC_LENGTH = 400

save_model = False
load_model = False


if socket.gethostname() == 'jacob':
    path = '/home/jacob/Desktop/DeepLearning_summarization/Data/train_medium_unique.csv'
    path_val = '/home/jacob/Desktop/DeepLearning_summarization/Data/validation_medium_unique.csv'
    PATH = '/home/jacob/Desktop/DeepLearning_summarization/'
else:
    path = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/train_medium_unique.csv'
    path_val = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/validation_medium_unique.csv'
    PATH = '/work3/s172727/DeepLearning_summarization/saved_network'
glove_dim = 50

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
#print("device: ", device, flush=True)
#os.environ['CUDA_LAUNCH_BLOCKING']='1' 
os.environ['THC_CACHING_ALLOCATOR']='1' #avoid CPU synchronizations https://github.com/torch/cutorch
#TODO: How do we enable sort in dataloader?


#%%
"""
Data loader part
"""
if dataset_type == 'dummy':
    TEXT = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
    LABEL = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
    train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
    validation_set = data.TabularDataset(path_val, 'CSV', fields=[('data', TEXT), ('label', LABEL)], skip_header=True ) 
    
    TEXT.build_vocab(train_set, vectors="glove.6B."+str(glove_dim)+"d")
    LABEL.vocab = TEXT.vocab
    vocab = copy.deepcopy(TEXT.vocab)

#GloVe embedding function

#This is the glove embedding for every word in vocab, we dont need to load it into memory
#w = embed.weight.data.copy_(vocab.vectors)

    #Data loader iterator
    dataset_iter = data.Iterator(train_set, batch_size=BATCH_SIZE, device=device,
            train=True, shuffle=False, repeat=False, sort=True, sort_key=lambda x: -len(x.data))
    
    dataset_iter_val = data.Iterator(validation_set, batch_size=1, device=device,
            train=True, shuffle=True, repeat=False, sort=False)
else:
    
    
    FIELD = ReversibleField(batch_first=False, init_token='<init>', eos_token='<eos>', lower=True, include_lengths=True)
    
    split_cnn = CNN.splits(fields=FIELD)
    split_dm = DailyMail.splits(fields=FIELD)
    
    for scnn, sdm in zip(split_cnn, split_dm):
        scnn.examples.extend(sdm)
    split = split_cnn
    
    FIELD.build_vocab(split[0].src, vectors="glove.6B."+str(glove_dim)+"d")
    vocab = copy.deepcopy(FIELD.vocab)
    
    dataset_iter, dataset_iter_val, dataset_iter_test = BucketIterator.splits(split, batch_size=BATCH_SIZE, sort=True, sort_key=lambda x: -len(x.src), device=device)

embed_glove = torch.nn.Embedding(vocab_size, glove_dim, sparse=True).to(device)
glove_weights = vocab.vectors[:vocab_size].cuda()
embed_oov = torch.nn.Embedding(len(vocab), glove_dim, sparse=True).to(device)
embed_oov.weight.data *=torch.cat(((torch.sum(embed_glove.weight.data,1) ==0).unsqueeze(1).expand_as(embed_glove.weight.data).float(), torch.ones(len(vocab) - vocab_size, glove_dim, device=device)))


embed = torch.nn.Embedding(len(vocab), glove_dim, sparse=True).to(device)
embed.weight.data = torch.cat((embed_glove.weight.data, torch.zeros(len(vocab) - vocab_size, glove_dim, device=device))) + embed_oov.weight.data
embed.weight.requires_grad = False

"""
Mask functions
"""

def label_mask(y):
    """y is the label tensor (no glove embedding)
       returns the mask to use for negating the padding"""
    mask = (y[1:] != 1)
    return mask.float()

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

def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)
     
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

                          
    def forward(self, x, batch_size, hidden, cell_state):
        output_enc, (hidden_enc, cell_state_enc) = self.lstm(x, (hidden, cell_state))
        return output_enc, (hidden_enc, cell_state_enc)

    def initHidden(self, batch_size, train=True):
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
        self.pgen_linear = nn.Linear(2*hidden_size + hidden_size + glove_dim, 1, bias=True)

#    def forward(self, output_enc, real_index, coverage, input_dec, hidden_enc, cell_state_enc, att_mask, vocab_ext, first_word=True):
    def forward(self, output_enc, real_index, coverage, input_dec, hidden_enc, cell_state_enc, att_mask, input_length, big_vocab_size, batch_size, first_word=True):
        #During training input_dec should be the label (sequence), during test it should the previous output (one word)
        #We reduce the forwards and backwards direction hidden states into
        #one, as our decoder is unidirectional.
#        print('start forward', int(torch.cuda.memory_allocated()/1000000))
#        t0 = time.time()
        if first_word:
            old_enc = torch.cat((hidden_enc[0:2],hidden_enc[2:]),dim=-1)
            old_cell = torch.cat((cell_state_enc[0:2],cell_state_enc[2:]),dim=-1)
            new_enc = self.reduce_dim(old_enc)
            new_cell = self.reduce_dim(old_cell)
            coverage = torch.zeros(output_enc.size()[0], output_enc.size()[1], 1, device=device) #TODO: check if correck and change to input_size
        else:
            new_enc = hidden_enc
            new_cell = cell_state_enc
        output_dec, (hidden_dec, cell_state_dec) = self.lstm(input_dec, (new_enc, new_cell))
#        print('lstm', int(torch.cuda.memory_allocated()/1000000))
        
#        t1 = time.time()
#        print('lstm', t1 - t0)
#        t0 = t1
        
#        pvocab = torch.zeros((len(output_dec), batch_size, vocab_size,int(torch.max(real_index)+1)))).cuda() 
        pvocab = torch.zeros(len(output_dec), batch_size, max(torch.max(real_index)+1, vocab_size), device=device)


        coverage_loss = torch.zeros(len(output_dec), batch_size, 1, device=device)
        
        pgen = torch.zeros(batch_size, 1, device=device)  
#        print('init over ', int(torch.cuda.memory_allocated()/1000000))
        
#        t1 = time.time()
#        print('init over', t1 - t0)
#        t0 = t1
        #Attention
        #We loop over t in the equation
#        t00 = time.time()
        for t in range(len(output_dec)):
            #Expand negates a loop over i
            #Output_dec contains all decoder states at times t=0, ... , t=N
            input_attention = torch.cat((output_enc, output_dec[t:t+1].expand(len(output_enc), batch_size, self.hidden_size), coverage), dim=2)
            e = self.attention(input_attention)
            e = self.tanh(e)
            e = self.attn_out(e)
            e = e + att_mask
            attention = self.softmax(e)
            
#            t1 = time.time()
#            print('attention', t1 - t00)
#            t0 = t1
#            print('attention', int(torch.cuda.memory_allocated()/1000000))
            
            coverage_loss[t] = torch.sum(torch.min(attention, coverage), dim=0)
            coverage = coverage + attention
#            print('coverage', int(torch.cuda.memory_allocated()/1000000))
#            t1 = time.time()
#            print('coverage', t1 - t0)
#            t0 = t1
            #Calculate context vector sum(a_i^t * h_i)
            #This becomes a weighted sum of hidden states, hidden states created
            #after inputing a word with high attention has more weight
            context = torch.sum(attention * output_enc, dim=0).reshape((1,batch_size, 2*self.hidden_size))
#            print('context', int(torch.cuda.memory_allocated()/1000000))         
#            t1 = time.time()
#            print('context', t1 - t0)
#            t0 = t1
            #Calculate pointer generation probability
            in_pgen = torch.cat((context, output_dec[t:t+1], embed(input_dec[t:t+1].long()).squeeze(2)),dim=2)
            
            pgen = self.pgen_linear(in_pgen)
            pgen = self.sigmoid(pgen)
            #pgen[t] = self.pgen_linear(in_pgen)
            #pgen[t] = self.sigmoid(pgen[t])
#            t1 = time.time()
#            print('pgen', t1 - t0)
#            t0 = t1
            
            #Calculate Pvocab (no softmax in training loop as it is included in cross entropy loss)
            p_vocab = torch.cat((output_dec[t:t+1],context),2)
            p_vocab = self.linearVocab1(p_vocab)
            p_vocab = self.linearVocab2(p_vocab)
            p_vocab = self.softmaxvocab(p_vocab)
            #Multiply pvocab with generation probability
            p_vocab = p_vocab * pgen
#            t1 = time.time()
#            print('p_vocab', t1 - t0)
#            t0 = t1
#            print('p_vocab', int(torch.cuda.memory_allocated()/1000000))
            #p_vocab is 1 x batch_size x vocab size
            if big_vocab_size > vocab_size:
                p_vocab = torch.cat((p_vocab, torch.zeros(1,batch_size,big_vocab_size-vocab_size, device=device)), dim=2)

            p_vocab_new = torch.zeros(big_vocab_size, batch_size, device=device)
            
#            print('initial memory: ', int(torch.cuda.memory_allocated()/1000000))
#            t0 = time.time()
#            pointer_distrib.scatter_add_(0, real_index, attention.squeeze())
#            pointer_distrib_prob = torch.ger(1-pgen.squeeze(0).squeeze(1), pointer_distrib.squeeze().t().reshape(1,batch_size*big_vocab_size).squeeze()).reshape(batch_size, batch_size, big_vocab_size)
#            p_vocab_new = torch.diagonal(pointer_distrib_prob).t().unsqueeze(0)
#            t1 = time.time()
#            print('old_p_vocab_new', t1 - t0)
#            print('oldmemory: ', int(torch.cuda.memory_allocated()/1000000))
#            t0 = time.time()
#            print(torch.min(pgen))
            pgen.expand_as(attention)
            attention = (1 -pgen) * attention
            p_vocab_new.scatter_add_(0, real_index, attention.squeeze())
            p_vocab_new = p_vocab_new.reshape(1, batch_size, -1)
#            pgen_expanded = pgen.squeeze(2).expand(big_vocab_size, batch_size)
#            smaller_p_vocab_new = ((1 - pgen_expanded.reshape(-1)) * pointer_distrib.reshape(-1)).reshape(1, batch_size, big_vocab_size)
#            t1 = time.time()
#            print('smaller_p_vocab_new', t1 - t0)
#            print('smallmemory: ', int(torch.cuda.memory_allocated()/1000000))
#            print(pointer_distrib.size())
#            print(p_vocab_new.size())
#            print(p_vocab_new - pointer_distrib)
#            print('p_vocab_new', int(torch.cuda.memory_allocated()/1000000))
#            t1 = time.time()
#            print('p_vocab_new', t1 - t0)
#            t0 = t1
            pvocab[t] = p_vocab_new + p_vocab
#            t1 = time.time()
#            print('pvocab', t1 - t0)
#            t0 = t1
#            print('pvocab', int(torch.cuda.memory_allocated()/1000000))
#        print('t', t)
#        print('big_vocab_size', big_vocab_size)
#        print('done', int(torch.cuda.memory_allocated()/1000000))
#        
#        t1 = time.time()
#        print('done loop', t1 - t00)
#        print('time per loop', (t1 - t00)/t)
        return pvocab, coverage_loss, coverage, (hidden_dec, cell_state_dec)
 #reduction='none' because we want one loss per element and then apply the mask
#def forward_pass(encoder, decoder, real_index, x, label, criterion, att_mask, vocab_ext):
def forward_pass(encoder, decoder, real_index, x, label, criterion, att_mask, input_length, big_vocab_size, batch_size):
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
#    print('start forward ', int(torch.cuda.memory_allocated()/1000000))
#    t0 = time.time()
    
    (hidden_enc, cell_state_enc) = encoder.initHidden(batch_size, train=True)
#    print('initHidden ', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('initHidden ', t1 - t0)
#    t0 = t1
    
    #Run encoder and get last hidden state (and output).
    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, batch_size, hidden_enc, cell_state_enc)
#    print('encoder forward ', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('encoder_forward', t1 - t0)
#    t0 = t1
    
#    output, cov_loss, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label[:-1], hidden_enc, cell_state_enc, att_mask, vocab_ext)
    output, cov_loss, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label[:-1], hidden_enc, cell_state_enc, att_mask, input_length, big_vocab_size, batch_size)
#    print('decoder forward ', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('decoder forward', t1 - t0)
#    t0 = t1
    
    label_hat = torch.argmax(output, -1)
#    print('label_hat', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('label_hat', t1 - t0)
#    t0 = t1
    
    output = output.permute([1,2,0]).unsqueeze(3) #N,C,d format where C number of classes for the Cross Entropy 
    label_ = label[1:].long().permute([1,2,0]).squeeze().unsqueeze(2)
#    print('output and label_', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('output and label_', t1 - t0)
#    t0 = t1
    
    loss = criterion(output, label_)
#    print('criterion', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('criterion', t1 - t0)
#    t0 = t1
    
    combined_loss = loss + torch.mul(cov_loss.permute([1,0,2]), LAMBDA_COVERAGE)
#    print('combined loss', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('combined loss', t1 - t0)
#    t0 = t1
    
    mask = label_mask(label).permute([1,0,2])
#    print('mask', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('mask', t1 - t0)
#    t0 = t1
    
    loss_mask = torch.sum(combined_loss * mask, dim=1) 
#    print('loss mask', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('loss mask', t1 - t0)
#    t0 = t1
    
    
    loss_batch = loss_mask / torch.sum(mask, dim=1)
#    print('loss batch', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('loss batch', t1 - t0)
#    t0 = t1
    
    total_loss = torch.mean(loss_batch)
#    print('total loss', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('total loss', t1 - t0)
#    t0 = t1
    
    loss_maskCE = torch.sum(loss * mask, dim=1) 
#    print('loss mask CE', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('loss mask ce', t1 - t0)
#    t0 = t1
    
    loss_batchCE = loss_maskCE / torch.sum(mask, dim=1)
#    print('loss batch CE', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('loss batch ce', t1 - t0)
#    t0 = t1
    
    total_lossCE = torch.mean(loss_batchCE)
#    print('total loss CE', int(torch.cuda.memory_allocated()/1000000))
#    t1 = time.time()
#    print('total loss ce', t1 - t0)
#    t0 = t1
    
    return output, total_lossCE, total_loss, label_hat

        
#def forward_pass_val(encoder, decoder, real_index, x, att_mask, vocab_ext):
def forward_pass_val(encoder, decoder, real_index, x, att_mask):
    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden(train=False)
    
    #Run encoder and get last hidden state (and output).

    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, hidden_enc, cell_state_enc)
    EOS = False
    label_hat = torch.zeros(1,1,1, device=device)
    label_hat[:] = 2
    out = []
#    output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label_hat, hidden_enc, cell_state_enc, att_mask, vocab_ext, first_word=True)
    output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label_hat, hidden_enc, cell_state_enc, att_mask, first_word=True)
    index = torch.argmax(output, -1)
    label_hat[:] = index
    if label_hat == 3:
        EOS = True
    out.append(index)
    while not EOS and len(out) < MAX_LENGTH:
#        output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, coverage, label_hat, hidden_dec, cell_state_dec, att_mask, vocab_ext, first_word=False)
        output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, coverage, label_hat, hidden_dec, cell_state_dec, att_mask, first_word=False)
        index = torch.argmax(output, -1)
        label_hat[:] = index
        if label_hat == 3:
            EOS = True
        out.append(index)
    return torch.stack(out)


#%%
#def train(encoder, decoder, data, criterion, enc_optimizer, dec_optimizer, epoch, train_loss):
def train(encoder, decoder, data, criterion, enc_optimizer, dec_optimizer, epoch):
    encoder.train()
    decoder.train()
    i = 0
    t0 = time.time()
    for batchData in data:
#        if i >=10:
#            break
#        vocab_ext = copy.deepcopy(vocab)
        if dataset_type == 'articles':
            real_index = batchData.src[0][:TRUNC_LENGTH,:]
            y1 = batchData.trg[0][:MAX_LENGTH]
        else:
            real_index = batchData.data[:TRUNC_LENGTH,:]
            y1 = batchData.label[:MAX_LENGTH]
        att_mask = attention_mask(real_index)
        y = torch.reshape(y1,(np.shape(y1)[0], np.shape(y1)[1], 1))
        y = y.float()
        x = embed(real_index)
        i += 1
        x = x.detach()
        y = y.detach()
        input_length = len(real_index)
        big_vocab_size = int(max(vocab_size, torch.max(real_index) + 1))
        batch_size = y.size()[1]
#        if np.bool(sum(torch.max(y.long(), dim=0)[0].squeeze() <= 
#                       torch.max(torch.cuda.LongTensor([vocab_size-1]*batch_size), 
#                                 torch.max(real_index, dim=0)[0].cuda())) != batch_size):
#            print("unknown word in target summary",flush=True)
#            continue
#        print(y.size())
#        x_oov = x * (x >= vocab_size).float()
#        mask = isin(y, x_oov.reshape(-1,1))
#        print(mask.size())
#        y = y * (mask + (y<big_vocab_size)).float()
#        print(y.size())
#        y = y.detach()
#        print(y.size())
        y = y * (y<big_vocab_size).float()

#        out, crossEnt_loss, loss, label_hat = forward_pass(encoder, decoder, real_index, x, y, criterion, att_mask, vocab_ext)
        out, crossEnt_loss, loss, label_hat = forward_pass(encoder, decoder, real_index, x, y, criterion, att_mask, input_length, big_vocab_size, batch_size)
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2)
        enc_optimizer.step()
        dec_optimizer.step()
#        print('CUDA memory usage: ', torch.cuda.max_memory_allocated(), ' out of ', torch.cuda.get_device_properties(0).total_memory, flush=True)
        if i % 20 == 0:
            print('Epoch {} [Batch {}]\tTraining loss: {:.4f} \tCoverage-CE ratio: :{:.4f}'.format(
                epoch, i, loss.item(), (loss.item() - crossEnt_loss)/crossEnt_loss),flush=True)
                
            t1 = time.time()
            print('time: {:.4f}\t memory: {}'.format(
                    t1 - t0, int(torch.cuda.max_memory_allocated()/1000000)), flush=True)
            t0 = t1
#            train_loss.append(float(loss.item()))
            
        if i % 60 == 0:
            #print('input')
            #print(display(real_index, vocab_ext))
            print('output',flush=True)
            try:
                ind = np.where(y1[1:,0] == 1)[0][0]
                print(display(label_hat[:ind,0], vocab),flush=True)
                print('real output',flush=True)
                print(display(y1[1:,0], vocab),flush=True)
            except:
                print(display(label_hat[:,0], vocab),flush=True)
                print('real output',flush=True)
                print(display(y1[1:,0], vocab),flush=True)
#            fig = plt.figure(figsize=(10,4))
#            plt.plot(range(len(train_loss)), train_loss, label='train_loss')
#            plt.legend()
#            plt.show()
#                print("Sorry, couldn't print it", flush=True)
#            print('CUDA memory usage: ', torch.cuda.max_memory_allocated(), ' out of ', torch.cuda.get_device_properties(0).total_memory, flush=True)


def validation(encoder, decoder, data):
    acc = []
#    vocab_ext = copy.deepcopy(vocab)
    i = 0
    for batchData in data:
        if i >= 2:
            break
        if dataset_type == 'articles':
            real_index = batchData.src[0]
            y = batchData.trg[0]
        else:
            real_index = batchData.data
            y = batchData.label
        real_index = real_index[:TRUNC_LENGTH,:]
        att_mask = torch.zeros(real_index.size()[0], real_index.size()[1], 1, device=device)
        y = torch.reshape(y,(np.shape(y)[0], np.shape(y)[1], 1))
        y = y.long()
        x = embed(real_index)
#        test_output = forward_pass_val(encoder, decoder, real_index, x, att_mask, vocab_ext)
        test_output = forward_pass_val(encoder, decoder, real_index, x, att_mask)
#        test_output = test_output.cuda()
        i += 1
        if len(test_output) == len(y[1:]):
            if len(test_output) == sum(test_output == y[1:]):
                acc.append(1)
            else:
                acc.append(0)
        else:
            acc.append(0)
    return acc

#%% Training op
if load_model:
    encoder = torch.load(PATH + '_encoder_articles2')
    decoder = torch.load(PATH + '_decoder_articles2')
else:
    encoder = BiEncoderRNN(glove_dim, hidden_size).to(device)
    decoder = BiDecoderRNN(1, hidden_size).to(device)
#enc_optimizer = optim.Adagrad(encoder.parameters(), lr=LEARNING_RATE, initial_accumulator_value=0.1)
#dec_optimizer = optim.Adagrad(decoder.parameters(), lr=LEARNING_RATE, initial_accumulator_value=0.1)
enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE)
#enc_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
#dec_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(reduction='none').cuda()

#%%
#train_loss=[]
try:
    for epoch in range(1, epochs + 1):
#        train(encoder, decoder, dataset_iter, criterion, enc_optimizer, dec_optimizer, epoch, train_loss)
        train(encoder, decoder, dataset_iter, criterion, enc_optimizer, dec_optimizer, epoch)
        torch.cuda.empty_cache()
        
#    acc = validation(encoder, decoder, dataset_iter_val)
        if save_model:
            torch.save(encoder, PATH + '_encoder_articles_hpc_epoch' + epoch)
            torch.save(decoder, PATH + '_decoder_articles_hpc_epoch' + epoch)
except KeyboardInterrupt:
    if save_model:
        torch.save(encoder, PATH + '_encoder_articles_hpc')
        torch.save(decoder, PATH + '_decoder_articles_hpc')

#except RuntimeError:
#    print('error')
#    print('CUDA memory usage: ', torch.cuda.max_memory_allocated(), ' out of ', torch.cuda.get_device_properties(0).total_memory, flush=True)
