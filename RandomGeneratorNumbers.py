#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:36:12 2018

@author: ubuntu
"""
from numpy import random

list_of_words = ['hello', 'the', 'a', 'and', 'no', 'do', 'be', 'help', 'because',
                 'is', 'monday', 'boy', 'girl', 'i', 'am', 'my', 'your', 'happy',
                 'count', 'random', 'words', 'green', 'field', 'nature', 'free',
                 'how', 'what', 'maybe', 'danish', 'copenhagen', 'maths', 'model',
                 'work', 'study', 'animal', 'dog', 'cat', 'bird', 'name', 'city',
                 'travel', 'country', 'fly', 'eat', 'sleep', 'think', 'find', 
                 'music', 'art', 'method', 'beautiful', 'smart', 'nice', 'blue',
                 'discover', 'write', 'read', 'are', 'to', 'for', 'why', 'play',
                 'in', 'out', 'from', 'or', 'either', 'paper', 'sport', 'like']

temp = [[word] * (i+1) for i, word in enumerate(list_of_words)]
weighted_list_of_words = [w for l in temp for w in l]

# maximum lenght of one row
def generate_seq(min_length, max_length, list_of_words):
    number_frequency = random.uniform(0.1, 0.9)
    length = random.randint(min_length, max_length)
    noNumber = True
    nwords=len(list_of_words)
    while noNumber:
        row = []
        numbers = []
        for k in range(length):
            u = random.random()
            if u >number_frequency:
                number = str(random.randint(20))
                row.append(number)
                numbers.append(number)
                noNumber = False
            else:
                word = list_of_words[random.randint(nwords)]
                row.append(word)
    return row, numbers

def save_to_file(N, filename, seed):
    # seed has to be 'train' or 'validation'
    if seed == 'train':
        random.seed(0)
    elif seed == 'validation':
        random.seed(1)
    outfile = open(filename, 'w')
    outfile.write('data,label\n')
    for i in range(N):
        input_text, target = generate_seq(10, 20, weighted_list_of_words)
        input_text = " ".join(input_text)
        target = " ".join(target)
        outfile.write(input_text + ',' + target + '\n')
    outfile.close()
    
save_to_file(10000, 'SampleData/train_short.csv', 'train')
save_to_file(2000, 'SampleData/validation_short.csv', 'validation')
