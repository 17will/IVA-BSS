import librosa
import numpy as np
import os
from sklearn.decomposition import PCA

def printA(fn):
    print("A")
    fn()
    print("OK")
    return "OK"

@printA
def printB():
    print("B")

print(printB)

PATH="D:\\Blind_Signal_Separation_Will\\MonauralSignalSeparation_WillWang_181106\\geniusturtle_1_01.wav"

y=librosa.load(PATH,sr=16000,mono=False)
y0=y[0]
WW=[[0.7,0.5],[0.4,0.8]]
WWW=PCA(n_components=2)
WWW.fit(WW)

print(0)