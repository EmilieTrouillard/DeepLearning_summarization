#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:36:12 2018

@author: ubuntu
"""

from numpy import random

# maximum lenght of one row
def generate_seq(min_length, max_length, list_of_words):
    length = random.randint(min_length, max_length)
    noNumber = True
    nwords=len(list_of_words)
    while noNumber:
        row = []
        numbers = []
        for k in range(length):
            u = random.random()
            if u > 0.8:
                number = str(random.randint(5000))
                row.append(number)
                numbers.append(number)
                noNumber = False
            else:
                word = list_of_words[random.randint(nwords)]
                row.append(word)
    return row, numbers