#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:36:12 2018

@author: ubuntu
"""

from numpy import random
import pandas as pd

file='words.txt'
words=pd.read_csv(file, header=None)
list_of_words = words[0].values
nwords=len(list_of_words)

output_file = open('text_with_numbers.txt', 'w') 
target_file = open('target_numbers.txt', 'w') 
# Numbers of rows to generate
N = 1000
# maximum lenght of one row
max_length = 20

for i in range(N):
    length = random.randint(max_length) + 1
    noNumber = True
    while noNumber:
        row = ''
        numbers = ''
        for k in range(length):
            u = random.random()
            if u > 0.8:
                number = str(random.randint(5000))
                row = row + number + ' '
                numbers = numbers + number + ' '
                noNumber = False
            else:
                word = list_of_words[random.randint(nwords)]
                row = row + word + ' '
    numbers = numbers[:-1]
    row = row[:-1]
    output_file.write(row + '\n')
    target_file.write(numbers + '\n')
    
output_file.close()
target_file.close()