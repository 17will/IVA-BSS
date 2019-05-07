import librosa
import numpy as np
import os
from sklearn.decomposition import PCA
#import pystoi
epsi=0.000001
MM=[[0.7,0.5],[0.4,0.8]]

class fivabss:
    def __init__(self, sr=16000, n_fft=1024, frameLen=1024, maxIter=1000, thre=0.000001, channel=2):
        self.sr=sr
        self.n_fft=n_fft
        self.frameLen=frameLen
        self.maxIter=maxIter
        self.thre=thre
        self.channel=channel
        #self.PATH=PATH
        #self.ls=ls

    def read_wave(self,wave_file):
        sr=self.sr
        n_fft=self.n_fft
        frameLen=self.frameLen
        channel=self.channel
        y,_=librosa.load(wave_file,sr,mono=False)
        y_mixed=self.linear_mixing(y)
        
        tmp=(librosa.stft(y_mixed[0,:],n_fft,hop_length=frameLen//2,win_length=frameLen))
        ys=np.zeros((channel,)+np.shape(tmp),dtype=np.complex)
        ys[0,:]=tmp
        for i in range(1,channel):
            ys[i,:]=(librosa.stft(y_mixed[1,:],n_fft,hop_length=frameLen//2,win_length=frameLen))
        
        return ys,len(y[0,:])

    def generator(self,PATH,step):
        files=os.listdir(PATH)
        filesNum=len(files)
        for i in range(0,filesNum,step):
            if PATH=="." or PATH=="":
                yield files[i],files[i]
            else:
                yield files[i],PATH + "\\" + files[i]

    def batched_processing(self,PATH,step):
        for wav_file,wave_file in self.generator(PATH,step):
            ys,y_len=self.read_wave(wave_file)
            ys_bss,_=self.bss(ys)
            self.write(ys_bss,y_len,wav_file)
            #TODO Validation using STOI and SNR

    #TODO change it to channel-changeable.
    def linear_mixing(self, y):
        y_shape=np.shape(y)
        MM=[[0.7,0.5],[0.4,0.8]]
        y_mixed=np.zeros(y_shape)
        y_mixed[0,:]=y[0,:]*MM[0][0]+y[1,:]*MM[1][0]
        y_mixed[1,:]=y[1,:]*MM[1][1]+y[0,:]*MM[0][1]
        #filter bias
        y_mixed[0,:]=y_mixed[0,:]-np.mean(y_mixed[0,:])
        y_mixed[1,:]=y_mixed[1,:]-np.mean(y_mixed[1,:])

        return y_mixed

    #TODO conv_mixing
    
    def nonlinear_function(self,fun,A):
        return fun(A)

    def bss(self, ys):
        ys_shape=np.shape(ys)
        N=ys_shape[2]
        nfreq=ys_shape[1]
        Wp_shape=(self.channel,self.channel,nfreq)
        Q=np.zeros(Wp_shape,dtype=np.complex)
        Wp=np.zeros(Wp_shape,dtype=np.complex)
        W=np.zeros(Wp_shape,dtype=np.complex)
        Xp=np.zeros(ys_shape,np.complex)
        S=np.zeros(ys_shape,np.complex)
        pObj=float('inf')
        S=np.zeros(ys_shape,np.complex)
        #PCA process
        for k in range(nfreq):
            # tmp_pca=PCA(n_components=2)
            # tmp_pca.fit(ys[:,k,:])
            
            tmp=np.zeros((self.channel,1))
            tmp[:,0]=np.mean(ys[:,k,:],1)
            Xmean=np.dot(tmp,np.ones((1,N)))
            Rxx=np.dot((ys[:,k,:]-Xmean),np.transpose(ys[:,k,:]-Xmean))/N
            d,E=np.linalg.eig(Rxx)
            D=np.diag(d**(0.5))
            D=np.linalg.inv(D)

            Q[:,:,k]=np.dot(D,np.transpose(E))
            Xp[:,k,:]=np.dot(Q[:,:,k],(ys[:,k,:]-Xmean))

            Wp[:,:,k]=np.eye(self.channel)
        
        #Iteration
        for iter in range(maxIter):
            for k in range(nfreq):
                S[:,k,:]=np.dot(Wp[:,:,k],Xp[:,k,:])

            S2=(np.abs(S))**2
            Ssq = (np.sum(S2,1))**0.5
            Ssq1=(Ssq+epsi)**(-1)
            Ssq3=self.nonlinear_function(xpower3,Ssq1)

            for k in range(nfreq):
                
                #Calculating Hessian Matrix and using non-linear function
                Zta=np.diag(np.mean((Ssq1-Ssq3*S2[:,k,:]),1))
                Phi=Ssq1*S[:,k,:]

                #Updating unmixing matrices
                Wp[:,:,k]=np.dot(Zta,Wp[:,:,k])-np.dot(Phi,np.transpose(Xp[:,k,:]))/N

                #Decorrelation
                Wp[:,:,k]=np.dot(matrixPowerMinusHalf(np.dot(Wp[:,:,k],np.transpose(Wp[:,:,k]))),Wp[:,:,k])

                

            Obj=np.sum(np.sum(Ssq)/(N*self.channel*nfreq))
            dObj=(pObj-Obj)/np.abs(Obj)
            pObj=Obj

            if iter%10==0:
                print("%d iterations: Objective=%e, dObj=%e\n"%(iter,Obj,dObj))

            if abs(dObj)<self.thre:
                break


        #Recover to stft-domain
        for k in range(nfreq):
            W[:,:,k]=np.dot(Wp[:,:,k],Q[:,:,k])
            teemp=np.diag(np.linalg.pinv(W[:,:,k]))
            W[:,:,k]=np.dot(teemp,W[:,:,k])
            
            #TODO spectral smoothing
            
            S[:,k,:]=np.dot(W[:,:,k],ys[:,k,:])
        
        return S,W

    def write(self, ys_bss, y_len, wave_file):
        ys_after=np.zeros((self.channel,y_len))
        
        ys_after[0,:]=librosa.istft(ys_bss[0,:,:],hop_length=self.frameLen//2,win_length=self.frameLen,length=y_len)
        ys_after[1,:]=librosa.istft(ys_bss[1,:,:],hop_length=self.frameLen//2,win_length=self.frameLen,length=y_len)
        librosa.output.write_wav("separated\\Processed_"+wave_file,ys_after,sr=self.sr,norm=True)

        #return ys_after


def xpower3(A):
        return A**3


    # @nonlinear_function
def sigmoid(A):
    return 1.0/(1+np.exp(-A))

    # @nonlinear_function
def tanh(self,A):
    return np.tanh(A)


def matrixPowerMinusHalf(A):
    """
    求矩阵的-1/2次方
    """
    # v 为特征值    Q 为特征向量
    v, Q = np.linalg.eig(A)
    # print(v)
    V = np.diag(v**(-0.5))
    # print(V)
    T = Q * V * Q**-1

    return T

PATH="D:\\Blind_Signal_Separation_Will\\new_start\\music-source-separation\\dataset\\mir-1k\\Wavfile"
sr=16000
frameLen=1024
n_fft=1024
maxIter=1000
thre=0.000001
Experiment=fivabss(sr,n_fft,frameLen,maxIter,thre)
Experiment.batched_processing(PATH,10)
print(0)