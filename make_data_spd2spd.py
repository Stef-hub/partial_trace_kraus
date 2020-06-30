#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Generates various toy data for PSD to PSD experiments

import sys
import numpy as np
import torch
from util import build_random_A, build_X, build_Y

noise = 0.1 


device = torch.device('cpu') #('cuda')

t = 1
n, n_test = int(sys.argv[1]), 1000
m = int(sys.argv[2])  # kraus rank 
d, p = int(sys.argv[3]), int(sys.argv[4])

nb_trial = 10

for i in range(nb_trial):
    
    A = build_random_A(m,p,d,device = device)
    X = build_X(n,d,device)
    Y = build_Y(n,X,A,noise = noise,device = device)
    Xt = build_X(n_test,d,device)
    Yt = build_Y(n_test,Xt,A,noise = noise,device = device)

    np.save('data/spd2spd/X_'+str(n)+'_'+str(d)+'_'+str(p)+'_'+str(m)+'_'+str(i)+'.npy', X)
    np.save('data/spd2spd/Y_'+str(n)+'_'+str(d)+'_'+str(p)+'_'+str(m)+'_'+str(i)+'.npy', Y)
    np.save('data/spd2spd/Xt_'+str(n)+'_'+str(d)+'_'+str(p)+'_'+str(m)+'_'+str(i)+'.npy', Xt)
    np.save('data/spd2spd/Yt_'+str(n)+'_'+str(d)+'_'+str(p)+'_'+str(m)+'_'+str(i)+'.npy', Yt)
