# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:03:47 2022

@author: ariasvts
"""

import sys
sys.path.append('./praat')
import prosody as pr
import scipy as sp
import fundamental_freq as f0z
import numpy as np
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
# from math import acos, degrees


def compute_vrt(signal,fs,slope_i=0.2,slope_t=0.8,win_time=0.02,step_time=0.01,segment=0.8,method='RMS'):
    """

    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #Signal normalization
    sig = signal-np.mean(signal)
    sig = sig/np.max(sig)
    #----
    lseg = int(segment*fs)
    time_offset = 0.05#Add X ms at the beginning
    pre_vo = int(time_offset*fs)#To set new beginning of the signal
    #Pad zeros to ensure that the signal is sufficiently long for analysis.
    # padd = np.zeros(pre_vo)
    padd = np.random.normal(-0.001,0.001,pre_vo)
    sig = np.insert(sig,0,padd)
    # padd = np.zeros(lseg)
    padd = np.random.normal(-0.001,0.001,lseg)
    sig = np.insert(sig,len(sig),padd)
    #-
    #RMS envelope
    if method =='RMS':
        envelope = rms(sig,fs,win_time,step_time)
        envelope = np.insert(envelope,0,np.zeros(int(win_time/(2*step_time))))   
        fs = 1/step_time     
        lseg = int(segment*fs)#Recompute to adjust
    #-
    #Hilbert envelope
    elif method=='Hilbert':
        envelope,phase,tfs = hilb_tr(sig,fs,glen = 0.08)
    vo = np.where(envelope>=(0.01*np.max(envelope)))[0][0]#voice initiation
        
    envelope = envelope/np.max(np.abs(envelope))
    envelope = envelope[0:lseg+vo]
    #-
    ti = np.where(envelope>=(slope_i*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    ti = ti -time_offset
    ti = np.round(ti,3)
    tf = np.where(envelope>=(slope_t*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    tf = tf -time_offset
    tf = np.round(tf,3)
    #-
    envelope = envelope[pre_vo:]
    #Compute the inclination of the rising onset
    #Normalize the points that form the right triangle
    x1 = ti/(len(envelope)/fs)#Normalized time
    x2 = tf/(len(envelope)/fs)
    # x1 = ti#Normalized time
    # x2 = tf
    y1 = envelope[int(ti*fs)]
    y2 = envelope[int(tf*fs)]
    #Compute the vertices
    hyp = euc_dist(x1,y1,x2,y2)#Hypotenuse
    adj = euc_dist(x1,y1,x2,y1)#Adjacent
    #Angle between hypothenuse and adjacent
    # theta = np.arccos(adj/hyp)
    theta = np.arctan((y2-y1)/(x2-x1))
    # Phase in degrees
    slope = theta*180/np.pi
    #-
    #Area triangle rise time
    opp = np.sqrt((hyp**2)-(adj**2))
    area = 0.5*(adj*opp)
    #Slope using polynomial fit
    # coeff = np.poly1d(np.polyfit(np.arange(0,int(tf*fs)-int(ti*fs)), envelope[int(ti*fs):int(tf*fs)],1))
    # slope = coeff[0]
    # xp = np.linspace(0, int(tf*fs)-int(ti*fs),int(tf*fs)-int(ti*fs))
    # rslo = coeff(xp)
    #-
    rt = tf-ti
    Results = {'VRT':rt,
               'Slope':slope,
               'Area':area,
                'Envelope':envelope,
               # 'Envelope':phase,
               # 'd1Env':denv,
               # 'd2Env':ddenv,
               'Signal':sig,
               'Start':ti,
               # 'Startd1':tid1,
               # 'Startd2':tid2,
               'End':tf}
    return Results
#*****************************************************************************
def compute_vrt_2(sig,fs,slope_i=0.2,slope_t=0.8,win_time=0.02,step_time=0.01,segment=0.8,method='RMS'):
    """

    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #Signal normalization
    sig = sig-np.mean(sig)
    sig = sig/np.max(sig)
    #----
    lseg = int(segment*fs)
    # pre_vo = int(0.05*fs)#To set new beginning of the signal
    #Pad zeros to ensure that the signal is sufficiently long for analysis.
    # padd = np.zeros(pre_vo)
    padd = np.random.normal(-0.001,0.001,pre_vo)
    sig = np.insert(sig,0,padd)
    # padd = np.zeros(lseg)
    padd = np.random.normal(-0.001,0.001,lseg)
    sig = np.insert(sig,len(sig),padd)
    #-
    envelope,phase,tfs = hilb_tr(sig,fs,glen = 0.08)
    vo = np.where(envelope>=(slope_i*np.max(envelope)))[0][0]#voice initiation
    if vo<pre_vo:
        sig = sig[0:vo+lseg]#Acoustic signal analized
        envelope = envelope[0:vo+lseg]#Use only 800ms from the onset initiation
        # phase = phase[0:vo+lseg]
    else:
        sig = sig[vo-pre_vo:vo+lseg]#Acoustic signal analized
        envelope = envelope[vo-pre_vo:vo+lseg]#Use only 800ms from the onset initiation
    #-
    ddenv = np.diff(envelope,2)#Derivative
    ddenv = ddenv/np.max(np.abs(ddenv))
    #-
    envelope = envelope/np.max(np.abs(envelope))
    #-
    ti = np.where(envelope>=(slope_i*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    ti = np.where(ddenv==np.max(ddenv[0:int(ti*fs)]))[0][0]/fs#Select the first frame that satisfy the condition
    tfHil = np.where(envelope>=(slope_t*np.max(envelope)))[0][0]/fs#Select the first frame that satisfy the condition
    #-
    #-
    f0 = pr.f0_contour_pr(sig,fs,sizeframe=win_time,step=step_time)
    #Compute F0
    frames = extract_windows(sig,int(win_time*fs),int(step_time*fs))
    H2 = np.zeros(frames.shape[0])#Maximum spectral amplitude H2
    for i in range(frames.shape[0]):
        #Get log-Magnitude spectrum
        spec = log_mag_spec(frames[i],fs,win_time,4096)
        H2[i] = get_spec_amp(spec, fs, 2*f0[i])    
    H2 = H2/np.max(np.abs(H2))
    
    tf = np.where(H2>=slope_t*np.max(H2))[0][0]*step_time#Select the first frame that satisfy the condition
    #-
    #Compute the inclination of the rising onset
    #Normalize the points that form the right triangle
    x1 = ti/(len(envelope)/fs)#Normalized time
    x2 = tf/(len(H2)*step_time)
    y1 = envelope[int(ti*fs)]
    y2 = H2[int(tf/step_time)]
    #Compute the vertices
    hyp = euc_dist(x1,y1,x2,y2)#Hypotenuse
    adj = euc_dist(x1,y1,x2,y1)#Adjacent
    #Angle between hypothenuse and adjacent
    # theta = np.arccos(adj/hyp)
    theta = np.arctan((y2-y1)/(x2-x1))
    # Phase in degrees
    slope = theta*180/np.pi
    #-
    #Area triangle rise time
    opp = np.sqrt((hyp**2)-(adj**2))
    area = 0.5*(adj*opp)
    #-
    rt = tf-ti
    Results = {'VRT':rt,
               'Slope':slope,
               'Area':area,
               'Envelope':envelope,
               'H2':H2,
               # 'd1Env':denv
               'Signal':sig,
               'Start':ti,
               # 'Startd1':tid1,
               # 'Startd2':tid2,
               'End':tf}
    return Results
#*****************************************************************************
def hilb_tr(signal,fs,smooth=True,glen = 0.01):
    """
    Apply hilbert transform over the signal to get
    the envelop and time fine structure
    
    If smooth true, then the amplitude envelope is smoothed with a gaussian window
    """
    #Hilbert Transform
    analytic_signal = hilbert(signal)
    #Amplitude Envelope
    amplitude_envelope = np.abs(analytic_signal)
    yreal = analytic_signal.real
    yimag = analytic_signal.imag
    phase_envelope = np.arctan(yimag.copy(),yreal.copy())
    #Temporal Fine Structure
    tfs = analytic_signal.imag/amplitude_envelope
    #Convolve amplitude evelope with Gaussian window
    if smooth==True:
        amplitude_envelope = smooth_curve(amplitude_envelope,fs,glen)
        phase_envelope = smooth_curve(phase_envelope,fs,glen)
    return amplitude_envelope,phase_envelope,tfs

#*****************************************************************************

def smooth_curve(x,fs,glen = 0.01):
    #Gaussian Window
    gauslen = int(fs*glen)
    window = gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal for smmothing
    smooth_x = x.copy()
    smooth_x = sp.convolve(smooth_x,window)
    smooth_x= smooth_x/np.max(smooth_x)
    ini = int(gauslen/2)
    fin = len(smooth_x)-ini
    x = smooth_x[ini:fin]
    return x

#****************************************************************************

def rms(sig,fs,win_time=0.025,step_time=0.01):
    """
    Sound Pressure Level as in:
        Švec JG, Granqvist S. Tutorial and Guidelines on Measurement of Sound 
        Pressure Level in Voice and Speech. Journal of Speech, Language, and Hearing Research. 
        2018 Mar 15;61(3):441-461. doi: 10.1044/2017_JSLHR-S-17-0095. PMID: 29450495.
        
    SPL = 20*log10(p/p0)
    
    20xlog refers to a root-power quantity e.g., volts, sound pressure, current...
    
    Intensity in dBs:
        ene = 10*log10(sum(x^2)/N)
    
    10xlog refers to a power quantity, i.e. quantities directly proportional to power
    
    x: speech signal
    N: lenght of x
    p = RMS value of x
    p0 = 20uPA = 0.00002 Hearing threshold
    """
    #Set a threshold based on the energy of the signal
    if len(sig)>3*int(win_time*fs):
        frames = extract_windows(sig,int(win_time*fs),int(step_time*fs))
    else:
        frames = list([sig])
    E = []
    p0 = 2*(10**-5)#Hearing threshold at SLP 0dB
    eps = np.finfo(np.float32).eps#To avoid errors
    for x in frames:
        #Sound Pressure Level (dBs)
        p = np.sqrt(np.sum((x)**2)/(len(x)))
        # Lp = 20*np.log10((p/p0)+eps)
        # Lp = 10*np.log10(p+eps)
        E.append(p)
    E = np.asarray(E)
    return E

#****************************************************************************

def extract_windows(signal, size, step,windowing = True):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    windows = np.vstack(windows)
    if windowing == True:
        windows = windows*np.hanning(size)
    return windows

#****************************************************************************

def euc_dist(x1,y1,x2,y2):
    return np.sqrt(((x2-x1)**2)+((y2-y1)**2))

#****************************************************************************
def get_spec_amp(S,fs,frqz,tol=0.1):
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
    
    try:
        #Find max spectral amplitude within tol% of the
        #frequency for a particular formant
        flow = xs[idxA]-(tol*xs[idxA])
        flow = int((flow*2*len(S))/fs)
        ftop = xs[idxA]+(tol*xs[idxA])
        ftop = int((ftop*2*len(S))/fs)
        A = np.max(S[flow:ftop])
    except:
        A = 0
    return A
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