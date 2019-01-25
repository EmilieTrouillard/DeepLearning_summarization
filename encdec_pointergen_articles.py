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
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
#import os
import matplotlib.pyplot as plt

if dataset_type == 'articles':
    from cnn_dm_torchtext_master.summarization import CNN, DailyMail

vocab_size = 50000 if dataset_type == 'articles' else 40 #Size of the vocab
BATCH_SIZE = 12 if dataset_type == 'articles' else 23  #Batch size
epochs = 1000 #How many epochs we train
attention_features = 100 #The number of features we calculate in the attention (Row amount of Wh, abigail eq 1)
vocab_features = 100 #The number of features we calculate when we calculate the vocab (abigail eq 4)
LEARNING_RATE = 0.0001
LAMBDA_COVERAGE = 1
layers_enc=1 #Num of layers in the encoder
layers_dec=1 #Num of layers in the decoder
MAX_LENGTH = 100
hidden_size = 256 #Hiddensize dimension (double the size for encoder, because bidirectional)
TRUNC_LENGTH = 400

save_model = True
load_model = False


if socket.gethostname() == 'jacob':
    path = '/home/jacob/Desktop/DeepLearning_summarization/Data/train_medium_unique.csv'
    path_val = '/home/jacob/Desktop/DeepLearning_summarization/Data/validation_medium_unique.csv'
    PATH = '/home/jacob/Desktop/DeepLearning_summarization/'
else:
    path = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/train_medium_unique_countries.csv'
    path_val = '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/SampleData/validation_medium_unique_countries.csv'
    PATH = '/work3/s172727/DeepLearning_summarization/saved_network' if dataset_type == 'articles' else '/media/ubuntu/1TO/DTU/courses/DeepLearning/DeepLearning_summarization/saved_network'
glove_dim = 50

device = torch.device("cuda:0")# if torch.cuda.is_available() else "cpu")
#print("device: ", device, flush=True)
#os.environ['CUDA_LAUNCH_BLOCKING']='1' 
#os.environ['THC_CACHING_ALLOCATOR']='1' #avoid CPU synchronizations https://github.com/torch/cutorch
#TODO: How do we enable sort in dataloader?


#%%
"""
Data loader part
"""
if dataset_type == 'dummy':
    TEXT = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True, include_lengths=True)
#    LABEL = data.Field(init_token='<bos>', eos_token='<eos>', sequential=True)
    train_set = data.TabularDataset(path, 'CSV', fields=[('data', TEXT), ('label', TEXT)], skip_header=True ) 
    validation_set = data.TabularDataset(path_val, 'CSV', fields=[('data', TEXT), ('label', TEXT)], skip_header=True ) 
    
    TEXT.build_vocab(train_set, vectors="glove.6B."+str(glove_dim)+"d")
#    LABEL.vocab = TEXT.vocab
    vocab = copy.deepcopy(TEXT.vocab)


#This is the glove embedding for every word in vocab, we dont need to load it into memory
#w = embed.weight.data.copy_(vocab.vectors)

    #Data loader iterator
    dataset_iter = data.Iterator(train_set, batch_size=BATCH_SIZE, device=device,
            train=True, shuffle=False, repeat=False, sort=True, sort_key=lambda x: len(x.data))
    
    dataset_iter_val = data.Iterator(validation_set, batch_size=1, device=device,
            train=True, shuffle=True, repeat=False, sort=False)
#else:
#    
#    
#    FIELD = ReversibleField(batch_first=False, init_token='<init>', eos_token='<eos>', lower=True, include_lengths=True)
#    
#    split_cnn = CNN.splits(fields=FIELD)
#    split_dm = DailyMail.splits(fields=FIELD)
#    
#    for scnn, sdm in zip(split_cnn, split_dm):
#        scnn.examples.extend(sdm)
#    split = split_cnn
#    
#    FIELD.build_vocab(split[0].src, vectors="glove.6B."+str(glove_dim)+"d")
#    vocab = copy.deepcopy(FIELD.vocab)
#    
#    dataset_iter, dataset_iter_val, dataset_iter_test = BucketIterator.splits(split, batch_size=BATCH_SIZE, sort=True, sort_key=lambda x: len(x.src), device=device)


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
            index = np.where(x[i]==1)[0][0] #<pad> tokens
            mask[i][index:] = -np.inf # we will add this mask, hence -inf
        except:
            pass
    return mask


def display(x, vocab_ext):
    return ' '.join([vocab_ext.itos[i] for i in x if vocab_ext.itos[i] != '<pad>'])

#def isin(ar1, ar2):
#    return (ar1[..., None] == ar2).any(-1)
     
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

                          
    def forward(self, x, x_length, batch_size, hidden, cell_state):
        lengths_list = x_length.view(-1).tolist()
        packed_emb = pack(x, lengths_list)
        output_enc, (hidden_enc, cell_state_enc) = self.lstm(packed_emb, (hidden, cell_state))
        output_enc = unpack(output_enc)[0]
        return output_enc, (hidden_enc, cell_state_enc)

    def initHidden(self, batch_size):
        #Initial state for the hidden state and cell state in the encoder
        #Size should be (num_directions*num_layers, batch_size, input_size)
        return (torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device), #h0
                torch.zeros(2*layers_enc, batch_size, self.hidden_size, device=device)) #c0


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
        if first_word:
            old_enc = torch.cat((hidden_enc[0:layers_enc],hidden_enc[layers_enc:]),dim=-1) #hidden_enc.size() = (2*layers_enc, batch_size, hidden_size)
            old_cell = torch.cat((cell_state_enc[0:layers_enc],cell_state_enc[layers_enc:]),dim=-1)
            new_enc = self.reduce_dim(old_enc)
            new_cell = self.reduce_dim(old_cell)
            coverage = torch.zeros(input_length, batch_size, 1, device=device) 
        else:
            new_enc = hidden_enc
            new_cell = cell_state_enc
            
        x_embed = embed(input_dec.long()).squeeze(2)
        output_dec, (hidden_dec, cell_state_dec) = self.lstm(x_embed, (new_enc, new_cell))

        pvocab = torch.zeros(len(output_dec), batch_size, big_vocab_size, device=device)

        coverage_loss = torch.zeros(len(output_dec), batch_size, 1, device=device)
        
        pgen = torch.zeros(batch_size, 1, device=device)  

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
            context = torch.sum(attention * output_enc, dim=0).unsqueeze(0)

            #Calculate pointer generation probability
            in_pgen = torch.cat((context, output_dec[t:t+1], embed(input_dec[t:t+1].long()).squeeze(2)),dim=2)
            pgen = self.pgen_linear(in_pgen)
            pgen = self.sigmoid(pgen)
                     
            #Calculate Pvocab 
            p_vocab = torch.cat((output_dec[t:t+1],context),2)
            p_vocab = self.linearVocab1(p_vocab)
            p_vocab = self.linearVocab2(p_vocab)
            p_vocab = self.softmaxvocab(p_vocab)
            #Multiply pvocab with generation probability
            p_vocab = p_vocab * pgen
            #p_vocab is 1 x batch_size x vocab size
            if big_vocab_size > vocab_size:
                p_vocab = torch.cat((p_vocab, torch.zeros(1,batch_size,big_vocab_size-vocab_size, device=device)), dim=2)
            
            p_vocab_pointer = torch.zeros(big_vocab_size, batch_size, device=device)
            
            pgen.expand_as(attention)
            attention = (1 -pgen) * attention
    
            p_vocab_pointer.scatter_add_(0, real_index, attention.squeeze(2))
            p_vocab_pointer = p_vocab_pointer.t().unsqueeze(0)
            pvocab[t] = torch.log(p_vocab_pointer + p_vocab)

        return pvocab, coverage_loss, coverage, (hidden_dec, cell_state_dec)
 
def forward_pass(encoder, decoder, real_index, x, label, x_length, criterion, att_mask, input_length, big_vocab_size, batch_size):
    #For training
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
   
    (hidden_enc, cell_state_enc) = encoder.initHidden(batch_size)
    
    #Run encoder and get last hidden state (and output).
    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, x_length, batch_size, hidden_enc, cell_state_enc)

    output, cov_loss, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label[:-1], hidden_enc, cell_state_enc, att_mask, input_length, big_vocab_size, batch_size)
    
    label_hat = torch.argmax(output, -1)
    
    output = output.permute([1,2,0]).unsqueeze(3) #N,C,d format where C number of classes for the Cross Entropy 
    label_ = label[1:].long().permute([1,2,0]).squeeze().unsqueeze(2)
    
    loss = criterion(output, label_)
    
    combined_loss = loss + torch.mul(cov_loss.permute([1,0,2]), LAMBDA_COVERAGE)
    
    mask = label_mask(label).permute([1,0,2])
    
    loss_mask = torch.sum(combined_loss * mask, dim=1)     
    
    loss_batch = loss_mask / torch.sum(mask, dim=1)
    
    total_loss = torch.mean(loss_batch)
    
    # Only cross entropy (CE) loss
    loss_maskCE = torch.sum(loss * mask, dim=1) 
    
    loss_batchCE = loss_maskCE / torch.sum(mask, dim=1)
    
    total_lossCE = torch.mean(loss_batchCE)

    return output, total_lossCE, total_loss, label_hat


def forward_pass_val(encoder, decoder, real_index, x, x_length, att_mask, input_length, big_vocab_size, batch_size):
    #greedy method
    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden(batch_size)
    
    #Run encoder and get last hidden state (and output).
    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, x_length, batch_size, hidden_enc, cell_state_enc)
    EOS = False
    label_hat = torch.zeros(1,1,1, device=device)
    label_hat[:] = 2 # <bos>
    out = []
    output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label_hat, hidden_enc, cell_state_enc, att_mask, input_length, big_vocab_size, batch_size, first_word=True)
    index = torch.argmax(output, -1)
    label_hat[:] = index
    if label_hat == 3:
        EOS = True
    out.append(index)
    while not EOS and len(out) < MAX_LENGTH:
        output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, coverage, label_hat, hidden_dec, cell_state_dec, att_mask, input_length, big_vocab_size, batch_size, first_word=False)
        index = torch.argmax(output, -1)
        label_hat[:] = index
        if label_hat == 3:
            EOS = True
        out.append(index)
    return torch.stack(out)


#%%
def beam_search_iteration(k, sequences, probabilities, decoder, coverages,
                          hidden_decs, cell_state_decs, output_enc, real_index, att_mask,
                          input_length, big_vocab_size, batch_size):
    new_sequences = []
    new_probabilities = []
    new_coverages = []
    new_hidden_decs = []
    new_cell_state_decs = []
    label_hat = torch.zeros(1,1,1, device=device)
    one_more_step = False
    for i, seq in enumerate(sequences):
        label_hat[:] = seq[-1] # take the last word of the sequence
        probability = probabilities[i] #the probability of that sequence
        if seq[-1] == 3: #<EOS>: if this is the last token, we keep that sequence as is
            new_sequences.append(seq)
            new_probabilities.append(probability)
            new_hidden_decs.append(hidden_decs[i])
            new_coverages.append(coverages[i])
            new_cell_state_decs.append(cell_state_decs[i])
        else: # otherwise, we generate the top k sequences seq + w
            one_more_step = True
            output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, coverages[i], label_hat, hidden_decs[i], 
                                 cell_state_decs[i], att_mask, input_length, big_vocab_size, batch_size, first_word=False)
            values, indices = torch.topk(output, k, dim=-1)
            indices = indices.squeeze().squeeze()
            values = values.squeeze().squeeze()
            
            for j in range(k):
                new_sequences.append(seq + [int(indices[j])])
                new_probabilities.append(probability*values[j])
                new_hidden_decs.append(hidden_dec)
                new_coverages.append(coverage)
                new_cell_state_decs.append(cell_state_dec)
    new_probabilities = torch.stack(tuple(new_probabilities))
    top_probabilities, ind = torch.topk(new_probabilities, k)
    top_sequences = [new_sequences[i] for i in ind]
    top_coverages = [new_coverages[i] for i in ind]
    top_hidden_decs = [new_hidden_decs[i] for i in ind]
    top_cell_state_decs = [new_cell_state_decs[i] for i in ind]
    return top_sequences, top_probabilities, top_coverages, top_hidden_decs, top_cell_state_decs, one_more_step    


def beam_search(k, encoder, decoder, real_index, x, x_length, att_mask, input_length, big_vocab_size, batch_size):
    iterations = 1
    #Reinitialize the state to zero since we have a new sample now.
    (hidden_enc, cell_state_enc) = encoder.initHidden(batch_size)
    #Run encoder and get last hidden state (and output).
    output_enc, (hidden_enc, cell_state_enc)=encoder.forward(x, x_length, batch_size, hidden_enc, cell_state_enc)
    label_hat = torch.zeros(1,1,1, device=device)
    label_hat[:] = 2
    out = []
    output, _, coverage, (hidden_dec, cell_state_dec) = decoder.forward(output_enc, real_index, None, label_hat, hidden_enc, cell_state_enc, att_mask, input_length, big_vocab_size, batch_size, first_word=True)
    probabilities, indices = torch.topk(output, k, dim=-1)
    indices = indices.squeeze().squeeze()
    probabilities = probabilities.squeeze().squeeze()
    sequences = [[int(i)] for i in indices]
    coverages = [coverage] * k
    hidden_decs = [hidden_dec] * k
    cell_state_decs = [cell_state_dec] * k
    one_more_step = True
    while one_more_step and iterations < MAX_LENGTH:
        sequences, probabilities, coverages, hidden_decs, cell_state_decs, one_more_step = beam_search_iteration(k, sequences, probabilities,
                                                                                                                 decoder, coverages, hidden_decs,
                                                                                                                 cell_state_decs, output_enc,
                                                                                                                 real_index, att_mask, input_length,
                                                                                                                 big_vocab_size, batch_size)
        iterations += 1
    highest_prob, idx = torch.max(probabilities, 0)
    out = sequences[idx]
    return torch.LongTensor(out).unsqueeze(1).unsqueeze(1)
    

#%%
def train(encoder, decoder, data, criterion, enc_optimizer, dec_optimizer, epoch, train_loss):
    encoder.train()
    decoder.train()
    i = 0
    for batchData in data:
#        if i >=2:
#            break
        if dataset_type == 'articles':
            real_index = batchData.src[0][:TRUNC_LENGTH,:]
            y1 = batchData.trg[0][:MAX_LENGTH]
            x_length = batchData.src[1]
        else:
            real_index = batchData.data[0][:TRUNC_LENGTH,:]
            y1 = batchData.label[0][:MAX_LENGTH]
            x_length = batchData.data[1]
        att_mask = attention_mask(real_index)
        y = y1.unsqueeze(2).float()
        x = embed(real_index)
        i += 1
#        x = x.detach()
#        y = y.detach()
        input_length = len(real_index)
        big_vocab_size = int(max(vocab_size, torch.max(real_index) + 1))
        batch_size = y.size()[1]
        y = y * (y<big_vocab_size).float()

        out, crossEnt_loss, loss, label_hat = forward_pass(encoder, decoder, real_index, x, y, x_length, criterion, att_mask, input_length, big_vocab_size, batch_size)
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2)
        enc_optimizer.step()
        dec_optimizer.step()
#        print('real output',flush=True)
#        print(display(y1[1:,0], vocab),flush=True)
#        print('real output',flush=True)
#        print(display(y1[1:,1], vocab),flush=True)
#        print('CUDA memory usage: ', torch.cuda.max_memory_allocated(), ' out of ', torch.cuda.get_device_properties(0).total_memory, flush=True)
        if i % 10 ==0:
            print('Epoch {} [Batch {}]\tTraining loss: {:.4f} \tCoverage-CE ratio: :{:.4f}'.format(
                epoch, i, loss.item(), (loss.item() - crossEnt_loss)/crossEnt_loss),flush=True)
#                
#            t1 = time.time()
#            print('time: {:.4f}\t memory: {}'.format(
#                    t1 - t0, int(torch.cuda.max_memory_allocated()/1000000)), flush=True)
#            t0 = t1
#            train_loss.append(float(loss.item()))
#            
        if i % 50 == 0:
#            fig = plt.figure(figsize=(10,4))
#            plt.plot(range(len(train_loss)), train_loss, label='train_loss')
#            plt.legend()
#            plt.show()
            #print('input')
            #print(display(real_index, vocab_ext))
            print('train output',flush=True)
            try:
                ind = np.where(y1[1:,0] == 1)[0][0]
                print(display(label_hat[:ind,0], vocab),flush=True)
                print('real output',flush=True)
                print(display(y1[1:,0], vocab),flush=True)
            except:
                print(display(label_hat[:,0], vocab),flush=True)
                print('real output',flush=True)
                print(display(y1[1:,0], vocab),flush=True)
            
            print('CUDA memory usage: ', torch.cuda.max_memory_allocated(), ' out of ', torch.cuda.get_device_properties(0).total_memory, flush=True)


def validation(encoder, decoder, data):
    encoder.eval()
    decoder.eval()
    i = 0
    for batchData in data:
        if dataset_type == 'articles':
            real_index = batchData.src[0][:TRUNC_LENGTH]
            y1 = batchData.trg[0][:MAX_LENGTH]
            x_length = batchData.src[1]
        else:
            real_index = batchData.data[0][:TRUNC_LENGTH]
            y1 = batchData.label[0][:MAX_LENGTH]
            x_length = batchData.data[1]
        att_mask = attention_mask(real_index)
        y = torch.reshape(y1,(np.shape(y1)[0], np.shape(y1)[1], 1))
        y = y.float()
        x = embed(real_index)
        i += 1
#        x = x.detach()
#        y = y.detach()
        input_length = len(real_index)
        big_vocab_size = int(max(vocab_size, torch.max(real_index) + 1))
        batch_size = y.size()[1]

        y = y * (y<big_vocab_size).float()
        label_hat_greedy = forward_pass_val(encoder, decoder, real_index, x, x_length, att_mask, input_length, big_vocab_size, batch_size)
        label_hat_beam = beam_search(10, encoder, decoder, real_index, x, x_length, att_mask, input_length, big_vocab_size, batch_size)
        print('output_greedy', flush=True)
        print(display(label_hat_greedy[:,0], vocab),flush=True)
        print('output_beam',flush=True)
        print(display(label_hat_beam[:,0], vocab),flush=True)
        print('real output',flush=True)
        print(display(y1[1:,0], vocab),flush=True)
        break

#%% Training op
if load_model:
    encoder = torch.load(PATH + '_encoder20')
    decoder = torch.load(PATH + '_decoder20')
    embed = torch.load(PATH + '_embed20')
else:
    embed_glove = torch.nn.Embedding(vocab_size, glove_dim, sparse=True).to(device)
    glove_weights = vocab.vectors[:vocab_size].cuda()
    embed_oov = torch.nn.Embedding(len(vocab), glove_dim, sparse=True).to(device)
    embed_oov.weight.data *=torch.cat(((torch.sum(embed_glove.weight.data,1) ==0).unsqueeze(1).expand_as(embed_glove.weight.data).float(), torch.ones(len(vocab) - vocab_size, glove_dim, device=device)))    
    embed = torch.nn.Embedding(len(vocab), glove_dim, sparse=True).to(device)
    embed.weight.data = torch.cat((embed_glove.weight.data, torch.zeros(len(vocab) - vocab_size, glove_dim, device=device))) + embed_oov.weight.data
    embed.weight.requires_grad = False
    encoder = BiEncoderRNN(glove_dim, hidden_size).to(device)
    decoder = BiDecoderRNN(glove_dim, hidden_size).to(device)
#enc_optimizer = optim.Adagrad(encoder.parameters(), lr=LEARNING_RATE, initial_accumulator_value=0.1)
#dec_optimizer = optim.Adagrad(decoder.parameters(), lr=LEARNING_RATE, initial_accumulator_value=0.1)
#enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE)
#dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE)
enc_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
dec_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
#enc_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
#dec_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)

#criterion = nn.CrossEntropyLoss(reduction='none').cuda()
criterion = nn.NLLLoss(reduction='none')
 #reduction='none' because we want one loss per element and then apply the mask

#%%
train_loss=[]
try:
    for epoch in range(1, epochs + 1):
        train(encoder, decoder, dataset_iter, criterion, enc_optimizer, dec_optimizer, epoch, train_loss)
#        if epoch % 50 ==0:
#        print('EPOCH ' + str(epoch))
#        fig = plt.figure(figsize=(10,4))
#        plt.plot(range(len(train_loss)), train_loss, label='train_loss')
#        plt.legend()
#        plt.show()
        torch.cuda.empty_cache()
#        validation(encoder, decoder, dataset_iter_val)
        if save_model:
            torch.save(encoder, PATH + '_encoder')
            torch.save(decoder, PATH + '_decoder')
            torch.save(embed, PATH + '_embed')
except KeyboardInterrupt:
    if save_model:
        torch.save(encoder, PATH + '_encoder20')
        torch.save(decoder, PATH + '_decoder20')
        torch.save(embed, PATH + '_embed20')
