'''
sampling frequency: 128 Hz
sampling period: 1/128 s
gender: 0--Male, 1--Female, 2--Unknown
tag: 1--normal, 0--abnormal
'''
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib as mtp
import scipy.signal as sig
import torch
import scipy.io as sio

def get_info(p_s,p_i,flag):
    dir_signal = os.listdir(p_s)
    dir_info = os.listdir(p_i)
    gender = {}
    age = {}
    signal = {}
    
    for i in range(len(dir_signal)):
        subp = p_s+dir_signal[i]
        raw = pd.read_table(subp)
        signal[i] = raw.to_numpy()[:,2]
        
    for i in range(len(dir_info)):
        subp = p_i+dir_info[i]
        raw = pd.read_table(subp)
        if flag:
            age[i] = round(float(raw.to_numpy()[0,0].split()[1]))
            gen = raw.to_numpy()[0,0].split()[3]
            if gen == 'F':
                gender[i] = 1
            elif gen == 'M':
                gender[i] = 0
            elif gen == '?':
                gender[i] = 2
        else:
            age[i] = round(float(raw.to_numpy()[0,0].split()[2]))
            gen = raw.to_numpy()[0,0].split()[4]
            if gen == 'F':
                gender[i] = 1
            elif gen == 'M':
                gender[i] = 0
            elif gen == '?':
                gender[i] = 2
    
    return signal,age,gender

root_path = 'data/total/'
# Load info for class_flag = 0 sample
p_n_s = root_path+'nsr_pure/'
p_n_i = root_path+'nsr_hd/'
n_signal, n_age, n_gender = get_info(p_n_s,p_n_i,0)
# Load info for class_flag =1 sample
p_c_s = root_path+'chf_pure/'
p_c_i = root_path+'chf_hd/'
c_signal, c_age, c_gender = get_info(p_c_s,p_c_i,1)

# dataset
D = []
for i in range(len(n_signal)):
    class_flag = 0
    d = (n_signal[i],n_age[i],n_gender[i],class_flag)
    D.append(d)

for i in range(len(c_signal)):
    class_flag = 1
    d = (c_signal[i],c_age[i],c_gender[i],class_flag)
    D.append(d)

sio.savemat('data.mat',{'d': D})