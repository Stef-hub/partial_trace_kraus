#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Genereates various toy data for PSD matrix completion experiments

import numpy as np
import matplotlib.pyplot as plt
import torch
from util import build_random_A, build_X, build_Y
import argparse


def completion_accuracy(orig, pred):

    if np.linalg.norm(pred, ord='fro') == 0:  #
        return 1 - np.trace(np.dot(orig, pred)) / np.linalg.norm(orig, ord='fro')

    return 1 - np.trace(np.dot(orig, pred))/(np.linalg.norm(orig, ord='fro')*np.linalg.norm(pred, ord='fro'))


def mat_to_vec(X, Y):
    n = X.shape[0]
    d = X.shape[1]
    p = Y.shape[1]
    Xnp = np.zeros((n, (d * d)))
    Ynp = np.zeros((n, (p * p)))
    for i in range(n):
        aux = X[i].numpy()
        Xnp[i] = np.reshape(aux, -1)
        aux = Y[i].numpy()
        Ynp[i] = np.reshape(aux, -1)
    return Xnp, Ynp


def vec_to_mat(Xv, Yv):
    n = Xv.shape[0]
    d = int(np.sqrt(Xv.shape[1]))
    p = int(np.sqrt(Yv.shape[1]))
    X = np.zeros((n, d, d))
    Y = np.zeros((n, p, p))
    for ii in range(n):
        aux = Xv[ii].detach().numpy()
        X[ii] = np.reshape(aux, (d, d), order='F')
        aux = Yv[ii].detach().numpy()
        Y[ii] = np.reshape(aux, (p, p), order='F')
    return X, Y


def vec_to_mat1(Yv):
    n = Yv.shape[0]
    pp = int(np.sqrt(Yv.shape[1]))
    Y = np.zeros((n, pp, pp))
    for ii in range(n):
        aux = Yv[ii]
        Y[ii] = np.reshape(aux, (pp, pp), order='F')
    return Y

# ================= parameters =================

parser = argparse.ArgumentParser()
parser.add_argument('-n', action='store', dest='n', default=24, type=int,
                    help='number of samples ')
parser.add_argument('-p', action='store', dest='p', default=7, type=int,
                    help='dataset type ')  # ok
parser.add_argument('-q', action='store', dest='q', default=4, type=int,
                    help='dataset type ')  # ok
parser.add_argument('-r', action='store', dest='r', default=10, type=int,
                    help='kraus rank ')  # ok
parser.add_argument('-m', action='store', dest='m', default=14, type=int,
                    help='dataset type ')

arguments = parser.parse_args()
n = arguments.n  # number of samples per class
p = arguments.p
q = arguments.q
r = arguments.r
m = arguments.m  # method of trou creation
# noise = arguments.s


# ================= get the data =================


if m < 3 and n%2 != 0:  # if can't make symmetric matrices with this n take one less training sample
    n = n-1


device = torch.device('cpu') #('cuda')
noise = 0.1
#r = 10

# experiments over various random matrices to be completed
for o in range(1):

    # block matrix to be completed, symmetric psd of rank r
    np.random.seed(o)
    
    A = build_random_A(r,q,p,device = device)
    XX = []
    for i in range(p):
      for j in range(p):
        mm = np.zeros((p,p))
        mm[i,j] = 1.
        XX.append(mm)
    XX = np.array(XX)
    XXX = torch.tensor(XX, device=device).float()
    YYY = build_Y(len(XXX),XXX,A,noise = noise,device = device)
    YYY = YYY.numpy()

    M = np.zeros((p*q, p*q))
    i = 0
    for p1 in range(p):
      for p2 in range(p):
        M[p1*q:(p1+1)*q, p2*q:(p2+1)*q] = YYY[i]
        i+=1

    if np.min(M) < 0: M += -2* np.min(M)
    np.save('data/completion/M_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'.npy', M)
    
    Y = M
    Ycorrect = M # for compatibility

    n_tot = p**2
    # n = 25
    t = n_tot - n
    print("ntot, n, t, p, q", n_tot, n, t, p, q)

    # put to right format
    X_all = torch.zeros((n_tot, p, p), device='cpu')
    Y_all = torch.zeros((n_tot, q, q), device='cpu')
    Y_all_correct = torch.zeros((n_tot, q, q), device='cpu')
    for p1 in range(p):
        for p2 in range(p):
            Y_all[p1*p+p2, :, :] = torch.from_numpy(Y[p1*q:(p1+1)*q, p2*q:(p2+1)*q])
            Y_all_correct[p1*p+p2, :, :] = torch.from_numpy(Ycorrect[p1*q:(p1+1)*q, p2*q:(p2+1)*q])
            tmp = np.zeros((p, p))
            tmp[p1, p2] = 1
            X_all[p1*p+p2, :, :] = torch.from_numpy(tmp)


    def get_mat_from_torch_array(ta, p, q):
        mat = np.zeros((p*q, p*q))
        # print(ta.shape, p, q)
        for p1 in range(p):
            for p2 in range(p):
                # print(ta[p1*p+p2, :, :].shape)
                # print(mat[p1*q:(p1+1)*q, p2*q:(p2+1)*q].shape)
                mat[p1*q:(p1+1)*q, p2*q:(p2+1)*q] = ta[p1*p+p2, :, :]  # todo rows and cols the other way around?
        return mat

    N = 6  # number of elements in lambda vector
    rank = m  # reduced rank regression

    lambda_vec = np.logspace(-3, 2, N)  # all right, I guess?
    print(lambda_vec)
    print()

    device = 'cpu'
    nb_iter = 5
    for ii in range(nb_iter):
   

        # # ----- just random -----
        np.random.seed(ii)  # always leave the random seed undommented for reproducible results!
        
        # any blocs can be removed, preserve symetry
        if m == 0:
          ordertmp = np.random.permutation(n_tot)
          # check for symetry!
          order = ordertmp[:int(n/2)].tolist()
          order2 = list(order)
          for o in order:
             ix = int(o / p)
             jx = int(o % p)
             order2.append(jx*p+ix)
          order2 = list(set(order2))
          if len(order2) < n:
            print("missing!")
            nn = len(order2)
            diag = np.arange(0, p)*(p+1)
            np.random.shuffle(diag)
            for _ in range(n-nn):
              iii = 0
              cont = True
              while iii < len(diag) and cont:
                if diag[iii] not in order2: 
                  order2.append(diag[iii])
                  cont = False
                iii = iii + 1
          order = np.array(order2)
          all = np.arange(0, p*p)
          others = np.array(list(set(all)-set(order)))
          order = np.append(order, others)

        # # ----- keep only diagonal blocs -----
        if m == 1:
          order = np.arange(0, p)*(p+1)
          n = len(order)
          all = np.arange(0, p*p)
          nondiag = np.array(list(set(all)-set(order)))
          order = np.append(order, nondiag)

        # ----- keep off-diagonal blocs -----
        if m == 2:
          order = np.arange(0, p)*(p+1)
          all = np.arange(0, p*p)
          trainingsamples = np.array(list(set(all)-set(order)))
          # order contains nondiag.. add missing elements here!
          while len(trainingsamples) > n:
            elem = np.random.choice(trainingsamples)
            [elemrow, elemcol] = np.unravel_index(elem, (p, p))
            elem2 = np.ravel_multi_index((elemcol, elemrow), (p, p))
            order = np.append(order, [elem, elem2])
            trainingsamples = np.array(list(set(all)-set(order)))
          order = np.append(trainingsamples, order)
        
        nm = n
        if m < 3:
            X_ = X_all[order[:nm], :, :]
            Xt_ = X_all[order[nm:], :, :]
            Y_ = Y_all[order[:nm], :, :]
            Yt_ = Y_all[order[nm:], :, :]

        # ------- create random missing entries not in blocs, but symmetric ---
        if m == 3:
          coord = []
          for i in range(p*q):
            for j in range(p*q):
              if j <= i: coord.append((i,j))
          np.random.shuffle(coord)
          coord = coord[:int(n/2)]
          coord2 = list(coord)
          for o in coord:
            ix = o[0]
            jx = o[1]
            coord2.append((jx,ix))
          coord2 = list(set(coord2))
          mask = np.zeros((p*q, p*q))
          for cc in coord2:
            mask[cc[0], cc[1]] = 1.

          # create training set, keep only blocs with enough data, currently at least 1. # 25%
          M_atrou = M * mask
          Y_all = torch.zeros((n_tot, q, q), device='cpu')
          trainset = []
          for p1 in range(p):
              for p2 in range(p):
                  if np.sum(mask[p1*q:(p1+1)*q, p2*q:(p2+1)*q]) > 0: # it's a bloc with enough entries to put in training set
                     trainset.append(p1*p+p2)
                  Y_all[p1*p+p2, :, :] = torch.from_numpy(M_atrou[p1*q:(p1+1)*q, p2*q:(p2+1)*q])
          order = np.arange(0, p*p)
          

          nm = len(order)
          X_ = X_all[trainset, :, :]
          Xt_ = X_all[order[:nm], :, :]
          Y_ = Y_all[trainset, :, :]
          Yt_ = Y_all[order[:nm], :, :]


        
        X = X_.numpy()
        Xt = Xt_.numpy()
        Y = Y_.numpy()
        Yt = Yt_.numpy()

        print(X.shape, Xt.shape)
        np.save('data/completion/X_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.npy', X_.numpy())
        np.save('data/completion/Xt_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.npy', Xt_.numpy())
        np.save('data/completion/Y_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.npy', Y_.numpy())
        np.save('data/completion/Yt_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.npy', Yt_.numpy())

        ## reconstruct complete M matrix
        mat = np.array(M)

        if m == 3: 
          mat_atrou = M_atrou
        else:
          mat_atrou = np.array(M)
          for j in range(Xt.shape[0]):
            c = np.where(Xt[j] == 1.)
            p1, p2 = int(c[0]), int(c[1])
            mat_atrou[p1*q:(p1+1)*q, p2*q:(p2+1)*q] = np.zeros(Yt[0].shape)

        cmap = plt.cm.pink
        vmin = float(np.min(mat))
        vmax = float(np.max(mat))

        plt.figure()
        plt.matshow(mat, fignum=False, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.savefig('data/completion/M_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.png')
        
        plt.figure()
        plt.matshow(mat_atrou, fignum=False, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.savefig('data/completion/M_atrou_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.png')
        
        np.save('data/completion/M_atrou_'+str(n)+'_'+str(p)+'_'+str(q)+'_'+str(r)+'_'+str(m)+'_'+str(ii)+'.npy', mat_atrou)

