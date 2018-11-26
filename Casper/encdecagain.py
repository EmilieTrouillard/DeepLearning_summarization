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
path = 'SampleData/train.csv'
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
        y = examples.label
        x = embed(x)        
        print(x)
        print(y[0])
        break
 
   
#We now have the data as x and the label as y
#Notice that a column in x corresponds to a training example.
#We can convert from data to string like this:
print()
print('The word corresponding to 28 is:', TEXT.vocab.itos[28]) #What is the word corresponding to 28?

#%%
#

# Define global variables
device='cpu'
seqlen=x.size(0)
embedsize=x.size(-1)
batch_size=x.size(1)
layers=2
blocks=30
maxlen=966#summary max length

def onehot(label, vocab,batch):
    onehot=torch.zeros(vocab,batch)
    for k in range(batch):
        onehot[label[k],k]=1
    return onehot
    
    
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
    def __init__(self, OutHidEnc, hidden_size, output_size):
        super(BiDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size =OutHidEnc, hidden_size=self.hidden_size,
                          bidirectional=True,num_layers=dlayers)
        
        self.out = nn.Linear(2*hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss= nn.CrossEntropyLoss()
    def forward(self, x, hidden,label):
        oui=[]
        losslist=[]
        output=0
        k=0
        EOS=False
        maxlab=len(label)
        while EOS==False and k<maxlab:
            output, hidden = self.lstm(x, hidden)
            output= self.out(output[-1])
            output = self.softmax(output)
            loss=nn.CrossEntropyLoss(output,onehot(label[k],vocab_size,batch_size))
            output=torch.argmax(output,-1)
            losslist.append(loss)
            oui.append(output)# [maxlen,batchsize]/[Word Position, batch position]
            output=label[k]
            k=k+1
        output=oui
        return output, losslist,hidden

    def initHidden(self):
        return (torch.zeros(2*dlayers,batch_size,self.hidden_size 
               , device=device),torch.zeros(2*dlayers,batch_size,self.hidden_size, device=device))
        
        
#%%        
enc=BiEncoderRNN(embedsize,blocks)
enc_h=BiEncoderRNN.initHidden(enc)
inp,hid=enc.forward(x,enc_h)
#fhid=hid[0][2,:,:]
#bhid=hid[0][3,:,:]
#decinp=torch.cat((fhid,bhid),dim=1)

#%% 
dlayers=layers
dblocks=blocks
decout=vocab_size
#decin=
dec=BiDecoderRNN(blocks ,dblocks,decout)
dec_h=BiEncoderRNN.initHidden(dec)
outp,loss,dechid=dec.forward(hid[0],dec_h,y)
#%%

enc=BiEncoderRNN(embedsize,blocks)
criterion=nn.CrossEntropyLoss()
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
    # Run encoder and get last hidden state (and output) 
    enc_h=BiEncoderRNN.initHidden(enc)
    enc_out,enc_h=enc.forward(x,enc_h)
    

    enc_h=BiEncoderRNN.initHidden(enc)
    inp,hid=enc.forward(x,enc_h)

    dec_h=BiEncoderRNN.initHidden(dec) # Init hidden state of decoder as hidden state of encoder
    dec_input = hid #Input to decoder; teacher forcing implement later.
    
    output,lossdec ,h= decoder(dec_input, dec_h,label)
    # Shape: [seqlen, batch,num_classes] with last dim containing log probabilities

    loss = criterion(output, t)
    output=torch.argmax(output, dim=-1)
    #accuracy = (output == t).type(torch.FloatTensor).mean()
    return output
#%%
i=forward_pass(enc,dec,x=x,label=y)

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
                100. * accuracy.item())



