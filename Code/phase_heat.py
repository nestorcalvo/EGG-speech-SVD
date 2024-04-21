# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:39:57 2023

@author: tomas
"""

import numpy as np
#-
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

#****************************************************************************

def get_phase_plots(signal,fs,win_time=0.04,step_time=0.01,windowing=False):
    """
    Get phase plots from a time series

    Parameters
    ----------
    signal : Time series
    fs : Sampling frequency
    win_time : Windows size in seconds. The default is 0.04.
    step_time : Step size in seconds. The default is 0.01.
    windowing : Frame the signal or not. The default is True.

    Returns
    -------
    heatmap : TYPE
        DESCRIPTION.

    """
    #Frame signals?
    if windowing:
        frames = extract_windows(signal, int(win_time*fs), int(step_time*fs))
        frames = frames*np.hanning(int(win_time*fs))
        heatmap = []#To save the images
        for f in frames:
            img = get_heat(f)
            heatmap.append(img)
    #Compute on the complete signal
    else:
        heatmap = get_heat(signal)
    #-
    return heatmap
    
#****************************************************************************

def get_heat(signal,plot=False):
    """
    Creates an image of a phase plot

    Parameters
    ----------
    signal: singal to be analyzed
    
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    heatmap : TYPE
        DESCRIPTION.

    """
    
    bins = 248
    s = 12#8#int(bins/128)#Resolution for the heatmap
    
    #Get analytical signal (Hilbert Transform)
    analytic_signal = hilbert(signal)
    #Envelope (Magnitude)
    env = np.absolute(analytic_signal)
    #-
    #Temporal structure signal
    tss = signal/env
    analytic_signal = hilbert(tss)
    #Real part (original signal)
    x = analytic_signal.real.copy()
    #Imaginary part (90° shift)
    y = analytic_signal.imag.copy()
    
    #Interpolation to make the plot smoother
    points = np.vstack([x,y]).T
    ixy = inter2D(points)
    x = ixy[:,0]
    y = ixy[:,1]
    #Get heatmap
    heatmap, xedges, yedges = np.histogram2d(x,y, bins=bins)
    #Include padding to avoid errors in the border due to gaussian filtering.
    pad = 8
    p = int(pad/2)
    hmap = np.zeros([bins+pad,bins+pad])
    hmap[p:-p:,p:-p:]  = heatmap
    heatmap = gaussian_filter(hmap, sigma=s)
    heatmap = heatmap.T
    #-    
    return heatmap


#****************************************************************************

def inter2D(points):
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    alpha = np.linspace(0, 1,5000)
    interpolator =  interp1d(distance, points, kind='cubic', axis=0)
    inter_points = interpolator(alpha)
    return inter_points

#****************************************************************************

def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)

#*****************************************************************************