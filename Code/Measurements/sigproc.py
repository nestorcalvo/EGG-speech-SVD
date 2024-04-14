"""
This file contains different useful funtions for signal processing
"""
import os
import numpy as np
import math
from scipy.io.wavfile import read #Leer y guardar audios
import scipy as sp
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert, gaussian
from scipy.signal import firwin,lfilter

def check_audio(audio_path):
    """
    Obtener frecuencia de muestreo y numero de canales
    Entrada
        :param audio_path: Carpeta que contiene los audios
    Salida
        :returns Frecuencia de muestreo promedio en la carpeta de audios
    """
    file_list = os.listdir(audio_path)
    list_fs = []
    for audio_name in file_list:
        fs,sig = read(audio_path+'/'+audio_name)
        list_fs.append(fs)
        channels = len(sig.shape)
        print('Audio: '+audio_name+' Fs: '+str(fs)+' Canales: '+str(channels))
    print('Fs Maximo: '+str(np.max(list_fs)))
    print('Fs Minimo: '+str(np.min(list_fs)))
    print('Fs promedio: '+str(np.mean(list_fs)))
    return np.mean(list_fs)

#=========================================================
def static_feats(featmat):
    """Compute static features
    :param featmat: Feature matrix
    :returns: statmat: Static feature matrix
    """
    mu = np.mean(featmat,0)
    st = np.std(featmat,0)
    ku = kurtosis(featmat,0)
    sk = skew(featmat,0)    
    statmat = np.hstack([mu,st,ku,sk])    
    return statmat.reshape(1,-1)

#============================================================
def min_max(x,a=0,b=1):
    """
    x = Array or matrix to normalize
    a = lower limit
    b = upper limit
    """
    if len(x.shape)==1:
        x = a+((x-np.min(x))*(b-a))/(np.max(x)-np.min(x))
    else:
        x = a+((x-np.min(x,0))*(b-a))/(np.max(x,0)-np.min(x,0))
    return x

#=========================================================
def add_noise(sig,target_snr_db=10):
    print('!'*50)
    print('Adding Gaussian noise of',target_snr_db,'dB to the signal')
    print('!'*50)
    #Remove DC level and re-scale between 1 and -1
    sig = norm_sig(sig)
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.sum(np.absolute(sig)**2)/len(sig)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig))
    # Noise up the original signal
    y_volts = sig + noise_volts
    return y_volts
#=========================================================
def norm_sig(sig):
    """Remove DC level and scale signal between -1 and 1.

    :param sig: Signal to normalize
    :returns: Normalized signal
    """
    #Eliminar nivel DC
    normsig = sig-np.mean(sig)
    #Escalar valores de amplitud entre -1 y 1
    normsig = normsig/float(np.max(np.absolute(normsig)))
    return normsig


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """    
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)
          
def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """    
    return 1.0/NFFT * np.square(magspec(frames,NFFT))
    
def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """    
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps
    
def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def highpass_fir(sigR,fs,fc,nfil):
    #sigR: Sennal a filtrar
    #fs: Frecuencia de muestreo de la sennal a filtrar
    #fc: Frecuencia de corte.
    #nfil: Orden del filtro
    largo = nfil+1 #  orden del filtro
    fcN = float(fc)/(float(fs)*0.5) # Frecuencia de corte normalizada
    #Filtro pasa bajas
    h = firwin(largo, fcN)
    #Inversion espectral para obtener pasa altas    
    h = -h
    h[int(largo/2)] = h[int(largo/2)] + 1
    #Aplicar transformada
    sigF = lfilter(h, 1,sigR)
    return sigF

#*****************************************************************************
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#*****************************************************************************
def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)
#*****************************************************************************
def get_spectrum(X,fs,win_time=0.025,step_time=0.01,n_padded=1024):
    """
    Compute log-power spectrum of a signal

    Parameters
    ----------
    X : Can be an array or a matrix. If X is an array, then it is assume that this
        is the signal. If X is a matrix, then it is assume that is the framed version
        of the signal
    fs : sampling frequency
    win_time : Duration of the analysis signal measured in seconds. The default is 0.025.
    step_time : Step size measured in seconds. The default is 0.01.
    n_padded : Resolution of the spectrum. The default is 1024.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #Convert from time to number of samples
    win_size = int(fs * win_time)
    step_size = int(fs * step_time)
    
    #If the signal is an array, then use windowing
    if X.ndim==1:
        X = extract_windows(X, win_size, step_size)
    
    # apply hanning window
    X *= np.hanning(win_size)

    # Fourier transform
    Y = np.fft.fft(X, n=n_padded)
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.log(np.abs(Y) ** 2+np.finfo(float).eps)
#*****************************************************************************
def powerspec(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.abs(Y) ** 2, n_padded
#*****************************************************************************
def powerspec2D(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
#    Y_real = np.abs(np.diff(Y.real,axis=1))
#    Y_img = np.abs(np.diff(Y.imag,axis=1))
#    Y_real = np.hstack([Y_real,Y_real[:,-1:]])
#    Y_img = np.hstack([Y_img,Y_img[:,-1:]])
    
    c = np.finfo(np.float).eps
    Y_real = np.abs(Y.real)+c
    Y_img = np.abs(Y.imag)+c
    mag = np.sqrt((Y_img)**2+(Y_real)**2)
    phase = np.arctan(Y_img.copy(),Y_real.copy())
    return mag,phase
#*****************************************************************************
def powerspec3D(X, rate, win_duration, n_padded_min=0):
    """
    Output: the power, magnitude, and phase spectrums of X
    """
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    power = np.abs(Y) ** 2
    Y_real = np.abs(Y.real)
    Y_img = np.abs(Y.imag)
    mag = np.sqrt((Y_img)**2+(Y_real)**2)
    phase = np.arctan(Y_img.copy(),Y_real.copy())
    return power,mag,phase
#********************************************************
def read_file(file_name):
    """
    Converts the text in a txt, txtgrid,... into a python list
    """
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    return lines
#****************************************************************************
def get_file(path,cond):
    """
    path: Folder containing the file name
    cond: Name, code, or number contained in the filename to be found:
          If the file name is 0001_ADCFGT.wav, cond could be 0001 or ADCFGT.
    """
    list_files = os.listdir(path)
    filesl = []
    for f in list_files:
        if f.upper().find(cond.upper())!=-1:
            filesl.append(f)
    if len(filesl)==1:
        f = [filesl[0]]
    elif len(filesl)>1:
        f = filesl
    else:
        f = ''#If no file is found, then return blank
    return f
#########################################
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
    
    #Temporal Fine Structure
    tfs = analytic_signal.imag/amplitude_envelope
    
    #Convolve amplitude evelope with Gaussian window
    if smooth==True:
        #Gaussian Window
        gauslen = int(fs*glen)
        window = gaussian(gauslen, std=int(gauslen*0.05))
        #Convolve signal for smmothing
        smooth_env = amplitude_envelope.copy()
        smooth_env = sp.convolve(amplitude_envelope,window)
        smooth_env = smooth_env/np.max(smooth_env)
        ini = int(gauslen/2)
        fin = len(smooth_env)-ini
        amplitude_envelope = smooth_env[ini:fin]
    return amplitude_envelope,tfs

def ACF_spec(sig,fs,filterbank,nfft=1024,win_time=0.04,step_time=0.01):
    """
    sig: Audio signal
    filterbank: As computed for the MFCC or Gammatone
    """
    frames = extract_windows(sig, int(win_time*fs), int(step_time*fs))
    frames *= np.hanning(int(win_time*fs))
    #Autocorrelation
    Rsig = np.zeros((frames.shape[0],int(nfft/2)+1))
    i = 0
    for fr in frames:
    
        ra = np.correlate(fr, fr, mode='full')
        ra = ra[int(ra.size/2):]#Only half
        ra/=np.max(ra)
    
        if len(ra)<int(nfft/2)+1:
            Rsig[i,0:len(ra)] = ra
        else:
            Rsig[i,:] = ra[0:int(nfft/2)+1]
    
        i+=1
    
    Rsig = np.vstack(Rsig)
    Rsig  = np.dot(Rsig ,filterbank.T)
    return Rsig