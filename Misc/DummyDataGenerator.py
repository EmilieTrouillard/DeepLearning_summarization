#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:36:12 2018

@author: ubuntu
"""
from numpy import random

# maximum lenght of one row
def generate_seq():
    indices = random.choice(range(10), 5, replace=False)
    L = [0] * 10
    labels = []
    for i in range(10):
        if i in indices:
            n = random.randint(5)
            L[i] = str(n)
            labels.append(str(n))
        else:
            n = random.randint(5, 10)
            L[i] = str(n)
    return L, labels

def save_to_file(N, filename, seed):
    # seed has to be 'train' or 'validation'
    if seed == 'train':
        random.seed(0)
    elif seed == 'validation':
        random.seed(1)
    outfile = open(filename, 'w')
    outfile.write('data,label\n')
    for i in range(N):
        input_text, target = generate_seq()
        input_text = " ".join(input_text)
        target = " ".join(target)
        outfile.write(input_text + ',' + target + '\n')
    outfile.close()
    
save_to_file(1000, 'DummyData/train.csv', 'train')
save_to_file(200, 'DummyData/validation.csv', 'validation')
