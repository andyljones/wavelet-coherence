#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:18:58 2017

@author: andyjones
"""

import pandas as pd
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

def load():
    path = 'data/j{}.txt'
    results = {}
    for name in ['ao', 'baltic']:
        results[name] = pd.read_csv(path.format(name), header=None, sep='\s+', index_col=0)[1]
    results = pd.concat(results, 1)
    results.index.name = ''
    
    return results

def scales_for(T):
    min_scale = 2
    max_scale = 1/3*T # dunno why .17
    interval = 1/12
    return 2**sp.arange(sp.log2(min_scale), sp.log2(max_scale), interval)

def wavelet(T, scale, k0=6):
    k = 2*sp.pi*sp.fftpack.fftfreq(T)
    nonnormed = sp.pi**-.25 * (k > 0) * sp.exp(-(scale*k - k0)**2 / 2)
    norm = sp.sqrt(2*sp.pi*scale)
    base = norm*nonnormed
    return base

def wavelength(scales, k0=6):
    return 4*sp.pi*scales/(k0 + sp.sqrt(2 + k0**2))

def pad(x):
    T = len(x)
    target = 2**sp.ceil(sp.log2(T))
    left = int((target - T)//2)
    right = int(target - T - left)
    padded = sp.r_[[0]*left, x, [0]*right]
    mask = sp.r_[[0]*left, [1]*len(x), [0]*right].astype(bool)
    return padded, mask
    
def wavelet_transform(x):
    padded, mask = pad(x.values)
    xhat = sp.fft(padded)
    T = len(padded)
    
    scales = scales_for(len(x))
    S = len(scales)
    
    output = sp.zeros((S, T), dtype=sp.complex64)
    for i, s in enumerate(scales):
        what = wavelet(T, scale=s)
        output[i] = sp.ifft(xhat*what)
        
    return pd.DataFrame(output[:, mask], wavelength(scales), x.index, dtype=sp.complex64)
        
def set_size():
    plt.gcf().set_size_inches(10, 10)

def plot_cart(x):
    plt.plot(sp.real(x))
    plt.plot(sp.imag(x))
    set_size()
    
def plot_spectrum(s):
    fig, ax = plt.subplots()
    ax.imshow(sp.absolute(s)**2, 
              interpolation='nearest', 
              cmap=plt.cm.viridis, 
              extent=(s.columns[0], s.columns[-1], s.shape[0], 0),
              aspect=1)
    

    ax.set_yticks(sp.arange(0, s.shape[0], 5))
    ax.set_yticklabels(['{:.0f}'.format(i) for i in s.index[::5]])
    set_size()