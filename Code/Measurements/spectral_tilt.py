# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:13:53 2021

@author: ariasvts
"""
# import sys
# sys.path.append('./utils')
import numpy as np
import prosody as pr
import lpc_formants as frqm
import fundamental_freq as f0z

def Spec_Tilt(sig,fs,win_time=0.05,step_time=0.01,nfft = 4096,nform = 3):
    win_size = int(win_time*fs)
    step_size = int(step_time*fs)
    
    #Compute number of short-time speech frames                
    if len(sig)>int(win_time*fs):
        n_frames = int((len(sig) - win_size) / step_size)
    else:
        n_frames = 1
        win_time = len(sig)/fs
    
    #Initialize variables
    formants = np.zeros([n_frames,nform])#To store the formants
    spec = np.zeros([n_frames,int(nfft/2)])#Log-magnitude spectrum
    # f0 = np.zeros(n_frames)#Fundamental frequency
    f0 = pr.f0_contour_pr(sig,fs,sizeframe=win_time,step=step_time)
    H1 = np.zeros(n_frames)#Maximum spectral amplitude H1
    H2 = np.zeros(n_frames)#Maximum spectral amplitude H2
    A2 = np.zeros(n_frames)#Maximum spectral amplitude A2
    A3 = np.zeros(n_frames)#Maximum spectral amplitude A3
    lpc_env = []
    for i in range(n_frames):
        #Get segment
        frame = sig[i * step_size : i * step_size + win_size].copy()
       
        #Windowing
        w = np.hanning(len(frame))
        frame = frame*w
       
        #Compute F0
        # f0[i] = f0z.get_f0_AC(frame, fs)
     
        #Get log-Magnitude spectrum
        spec[i,:] = log_mag_spec(frame,fs,win_time,nfft)
        
        #Get formants and LPC
        formants[i,:],lpcE = frqm.get_formants(frame,fs,meth='AC',nform=nform,pre_emph=False)
        
        #LPC envelope (No pre-emphasis)
        # _,lpcE  = frqm.get_formants(frame,fs,meth='Burg',nform=nform,pre_emph=False)
        lpc_env.append(lpcE)
        
        try:
            
            #Compute Harmonic ampltiudes (H1 and H2)       
            H1[i] = get_spec_amp(spec[i,:], fs, f0[i],0.2)
            H2[i] = get_spec_amp(spec[i,:], fs, 2*f0[i])
            
            #Compute Spectral amplitudes A2 and A3
            A2[i] = get_spec_amp(spec[i,:], fs, formants[i,1])
            A3[i] = get_spec_amp(spec[i,:], fs, formants[i,2])
        except:
            #Compute Harmonic ampltiudes (H1 and H2)       
            H1[i] = 0
            H2[i] = 0
            
            #Compute Spectral amplitudes A2 and A3
            A2[i] = 0
            A3[i] = 0
            
    
    uH1 = np.mean(H1)
    uH2 = np.mean(H2)
    uA2 = np.mean(A2)
    uA3 = np.mean(A3)
    formants[f0==0,0] = 0
    formants[f0==0,1] = 0
    
    #Compute features
    X = {'H1-H2':uH1-uH2,
         'H1-A2':uH1-uA2,
         'H1-A3':uH1-uA3,}
    #----------------------
    Param = {'f0 [Hz]': f0,
             'Formants [Hz]':formants,
             'Spectrum [dB]':spec,
             'LPC':np.asarray(lpc_env),
             'H1 [dB]':H1,
             'H2 [dB]':H2,
             'A2 [dB]':A2,
             'A3 [dB]':A3}
    return X,Param

#-----------------------------------------------------------------------------
def log_mag_spec(X, rate, win_duration, n_padded_min=0):
    """
    Compute the log-magnitude spectrum

    Parameters
    ----------
    X : Array or numpy matrix
        It can be either the speech signal or a matrix of speech frames
    rate : int
        Sampling frequency
    win_duration : TYPE
        Duration of X measured in seconds
    n_padded_min : Int power of 2, optional
        Number of bins for the FFT. The default is 0.

    Returns
    -------
    Array (or matrix) with the log-magnitude measured in dBs.

    """
    win_size = int(rate * win_duration)
    
    # apply hanning window
    # X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
    # Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
    
    # non-redundant part
    m = int(n_padded / 2)
    if len(Y.shape)==2:
        Y = Y[:, :m]
    else:
        Y = Y[:m] 
    return 20*np.log10(np.abs(Y))
#****************************************************************************
def get_spec_amp(S,fs,frqz,tol=0.1,xf=False):
    """
    Localize and extract the maximum spectral peak in the region of the 
    fundamental (H1, H2) or formant (A1,A2,A3) frequencies 

    Parameters
    ----------
    S : Log-magintude spectrum
    fs : Sampling frequency of the speech signal
    frqz : f0 (H1), 2*f0 (H2) or formant frequencies (A1,A2,A3))
    tol : Float. Tolerance value to set the range in which the 
                maximum spectral peak is localized
    xf: If True, an array with the frequency spectrum values (Hz) of S is returned.
    
    Returns
    -------
    A : Maximum spectral Amplitude
    xs : Array with frequency values

    """
    #Compute frequencies
    xs = np.linspace(0, int(fs/2), len(S))
        
    #Search for the formants in the FFT
    df = np.abs(xs-frqz)
    idxA = np.where(df==np.min(df))[0][0]
    
    #Find max spectral amplitude within tol% of the
    #frequency for a particular formant
    flow = xs[idxA]-(tol*xs[idxA])
    flow = int((flow*2*len(S))/fs)
    ftop = xs[idxA]+(tol*xs[idxA])
    ftop = int((ftop*2*len(S))/fs)
    A = np.max(S[flow:ftop])
    if xf==True:
        return A,xs
    else:
        return A
#****************************************************************************
          