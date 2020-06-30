#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time as time
import torch

def build_random_A(m,p,d, device = 'cpu'):    
    A = torch.randn(m,p,d, device = device)
    A = A/torch.max(torch.abs(A))
    return A

def build_X(n,d, device = 'cpu'):
    """ create n input covariance matrix of size d x d
        from random matrix
    """
    X = torch.zeros((n,d,d), device = device)
    for i in range(n):
        aux = torch.randn((d,d), device = device)
        X[i] = aux.t()@aux + torch.eye(d)*0.00
    return X


def build_Y(n,X,A, noise = 0 , device = 'cpu'):
    n = X.size(0)
    m = A.size(0)
    p = A.size(1)
    Y = torch.zeros((n,p,p), device = device)
    for i in range(n):
        for j in range(m):
            Y[i] += A[j]@X[i]@A[j].t()
    return Y
