# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:34:09 2021

@author: shovon5795
"""

import librosa
import librosa.display

import IPython.display as ipd

import matplotlib.pyplot as plt

'''filename = '1500-phrase.wav'
plt.figure(figsize = (12,5))
data, sample_rate = librosa.load(filename)
librosa.display.waveplot(data, sr = sample_rate)
ipd.Audio(filename)
'''
no_of_mfcc = 13

#mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc = no_of_mfcc)
#mfccs.shape

import glob
import os
import numpy as np
import pandas as pd

final_voice_dys =[]
col=0

pathd = r"E:\Research\SAM\Dataset\Dysphonie"

for filename in glob.glob(os.path.join(pathd, '*.wav')):
    datad, sample_rated = librosa.load(filename)
    mfccsd = librosa.feature.mfcc(y=datad, sr=sample_rated, n_mfcc=no_of_mfcc)
    mfccs1d = np.array(mfccsd)
    voice_dys = mfccs1d.flatten()
    final_voice_dys.append(voice_dys)

dataset1 = pd.DataFrame(final_voice_dys)
dysphon = dataset1.to_csv(r'E:\Research\SAM\Dataset\dysphon.csv', index = False)

pathh = r"E:\Research\SAM\Dataset\Healthy"
final_healthy =[]

for filename in glob.glob(os.path.join(pathh, '*.wav')):
    datah, sample_rateh = librosa.load(filename)
    mfccsh = librosa.feature.mfcc(y=datah, sr=sample_rateh, n_mfcc=no_of_mfcc)
    mfccs1h = np.array(mfccsh)
    voice_healthy = mfccs1h.flatten()
    final_healthy.append(voice_healthy)

dataset2 = pd.DataFrame(final_healthy)

healthy = dataset2.to_csv(r'E:\Research\SAM\Dataset\healthy.csv', index = False)

pathl = r"E:\Research\SAM\Dataset\Laryngitis"
final_lar =[]

for filename in glob.glob(os.path.join(pathl, '*.wav')):
    datal, sample_ratel = librosa.load(filename)
    mfccsl = librosa.feature.mfcc(y=datal, sr=sample_ratel, n_mfcc=no_of_mfcc)
    mfccs1l = np.array(mfccsl)
    voice_lar = mfccs1l.flatten()
    final_lar.append(voice_lar)

dataset3 = pd.DataFrame(final_lar)

lar = dataset3.to_csv(r'E:\Research\SAM\Dataset\laringitis.csv', index = False)


pathl = r"E:\Research\SAM\Dataset\Reinkes Edema"
final_rank =[]

for filename in glob.glob(os.path.join(pathl, '*.wav')):
    datar, sample_rater = librosa.load(filename)
    mfccsr = librosa.feature.mfcc(y=datar, sr=sample_rater, n_mfcc=no_of_mfcc)
    mfccs1r = np.array(mfccsr)
    voice_rank = mfccs1r.flatten()
    final_rank.append(voice_rank)

dataset4 = pd.DataFrame(final_rank)

rank = dataset4.to_csv(r'E:\Research\SAM\Dataset\ranke.csv', index = False)


final_dataset = pd.concat([dataset1, dataset2, dataset3, dataset4], axis = 0).to_csv(r'E:\Research\SAM\Dataset\final.csv', index = False)

df = pd.read_csv(r"E:\Research\SAM\Dataset\final.csv")

final_preprocessed_data = df.replace(np.nan, 0)
#final preprocessed_data.hist()
#plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_final_preprocessed_data = scaler.fit_transform(final_preprocessed_data) 
scaled_final_preprocessed_data = np.array(scaled_final_preprocessed_data)


final_preprocessed_data["Label"] = 0
final_preprocessed_data.iloc[0:52, 2769:2770] = 1  #Dysphonia
final_preprocessed_data.iloc[52:192, 2769:2770] = 0 #Healthy
final_preprocessed_data.iloc[192:315, 2769:2770] = 2 #Laryngitis
final_preprocessed_data.iloc[315:368, 2769:2770] = 3 #Rankei Odema

aggregated_dataset = final_preprocessed_data.to_csv(r"E:\Research\SAM\Dataset\final_version.csv")






    

