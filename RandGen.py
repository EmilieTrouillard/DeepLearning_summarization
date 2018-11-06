# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:36:27 2018

@author: caspe
"""

import numpy as np
import numpy.random as rand
from pandas import read_csv

file='words.txt'
words=read_csv(file)
nwords=words.shape[0]
space=' '
month=['January', 'February','March','April','May',
       'June', 'July','August', 'September', 
       'October', 'November', 'December']

def DateGen(inte):
    d=rand.randint(1,30)
    m=rand.randint(1,12)
    y=rand.randint(0,2018)
    date=' '
    if inte==1:
        date= '%.i/%.i/%.i' %(d,m,y) 
    if inte==2:
        date= '%.i/%.i/%.i' %(y,m,d) 
    if inte==3:
        date= '%.i/%.i/%.i' %(m,d,y) 
        
    elif inte==4 and 3<d: 
        date = 'the %.ith of ' %(d)
        date = date + month[m]
        s=', %.i' %(y)
        date = date+s
    else:
        if 1==d:
            date = 'the %.ist of ' %(d)
            date = date + month[m]
            s=', %.i' %(y)
            date = date+s
        elif d==2:
            date = 'the %.ind of ' %(d)
            date = date + month[m]
            s=', %.i' %(y) 
            date = date+s
        elif d==3:
            date = 'the %.ird of ' %(d)
            date = date + month[m]
            s=', %.i' %(y)   
            date = date+s
    date=space+date+space        
    return date
def StringGen(length):
    s=''
    ind=rand.randint(0,nwords,size=(1,length))
    for k in range(length):
        s=s+words.iloc[ind[0,k]][0]
        s=s+space
    return s    
def Inserter(date,string):
        r=rand.randint(0,len(string))
        ind=string.find(space,r)
        stri=string[0:ind]+date
        stri=stri+string[ind+1:]
        return stri
#%%
minlen=2
maxlen=4
probpct=0.2
nsent=100
df = pd.DataFrame(columns=['Sentence'])
dftarget = pd.DataFrame(columns=['Target'])
for k in range(0,nsent):
    d=DateGen(rand.randint(1,4))
    s=StringGen(rand.randint(minlen,maxlen))
    string=Inserter(d,s)
    if rand.rand()<probpct:
        ra=rand.randint(1,4)
        if ra==1:
            problem='/'
        elif ra==2:
            problem=' of '
        elif ra==3:
            problem= month[rand.randint(0,len(month))]
        elif ra==4:
            problem=' 2nd '
            
        string=Inserter(problem,string)
    df.loc[k,'Sentence']=string
    dftarget.loc[k,'Target']= d

np.savetxt('WriteTo.txt', df.values, fmt='%s')
np.savetxt('TargetSummaries.txt', dftarget.values, fmt='%s')


















