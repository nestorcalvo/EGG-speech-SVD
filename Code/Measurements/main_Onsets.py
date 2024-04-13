# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:47:22 2022

@author: ariasvts

IMPORTANT: The acoustic, egg, and airflow signal MUST have the same sampling frequency

"""
import logging
import numpy as np
from Measurements.Voice_onset import VocalAttackTime,VocalRiseTime, VoiceOnsetTime,vocal_rise_time


def getVoiceOnsets(data,time_stamps,opt,fs=16000):
    """
    Parameters
    ----------
    time_stamps: List with the time stamps of the segment to be analyzed
    
    opt : String
        String to select the Voice onset calculation method. The options are
        VAT: Computes the Vocal Attack Time
    

    Returns
    -------
    None.

    """
    sig_audio = None
    sig_egg = None
    sig_air = None
    #----------------------------
    if 'Acoustic mic' in data:
        sig_audio = data['Acoustic mic']['signal']
    #-
    if 'EGG' in data:
        sig_egg = data['EGG']['signal']
    #-
    if 'Airflow-Input A' in data:
        sig_air = data['Airflow-Input A']['signal']
    #----------------------------
    Onsets = {}
    #------------------------------
    #Vocal Attack Time
    VAT = VocalAttackTime.VocalAttackTime()
    #Vocal Onset Coordination. Same as VAT but acoustic is replaced with airflow
    VOC = VocalAttackTime.VocalAttackTime()
    #Voice Rise Time
    VRT = VocalRiseTime.VocalRiseTime()
    #Vowel Onset Time
    VOwT = VoiceOnsetTime.VoiceOnsetTime()
    f=1
    for i in time_stamps:
        #-
        ini_time = i[0,0]
        ini_sample = int(fs*ini_time)
        #-
        end_time = i[0,1]
        end_sample = int(fs*end_time)
        #------------------------------
        try:
            #------------------------------
            #Compute vocal attack time
            if (opt=='VAT')and(sig_audio is not None)and(sig_egg is not None):
                VAT.compute(sig_audio[ini_sample:end_sample],sig_egg[ini_sample:end_sample],fs)
                res = VAT.onset_label
                tm = get_segments(fs,res)[0]
                #Add the initial time of the segment that is being analyzed
                tm[0] = tm[0]+ini_time
                tm[1] = tm[1]+tm[0]
                Onsets['VAT_'+str(f)] = np.hstack([tm[0],tm[1]]).reshape(1,-1)
            #------------------------------
            #Compute vocal onset coordination. 
            if (opt=='VOC')and(sig_air is not None)and(sig_egg is not None):   
                VOC.compute(sig_air[ini_sample:end_sample],sig_egg[ini_sample:end_sample],fs)
                res = VOC.onset_label
                tm = get_segments(fs,res)[0]
                #Add the initial time of the segment that is being analyzed
                tm[0] = tm[0]+ini_time
                tm[1] = tm[1]+tm[0]
                Onsets['VOC_'+str(f)] = np.hstack([tm[0],tm[1]]).reshape(1,-1)
            #------------------------------
            #Compute Voice Rise Time
            if (opt=='VRT')and(sig_audio is not None):
                # VRT.compute(sig_audio[ini_sample:end_sample],fs)
                # res = VRT.onset_label
                # tm = get_segments(fs,res)[0]
                rt,ti,tf,envelope = vocal_rise_time.compute_vrt(sig_audio[ini_sample:end_sample], fs)
                tm = [ti,tf]
                #Add the initial time of the segment that is being analyzed
                tm[0] = tm[0]+ini_time
                tm[1] = tm[1]+tm[0]
                Onsets['VRT_'+str(f)] = np.hstack([tm[0],tm[1]]).reshape(1,-1)
            #------------------------------
            #Compute Vowel (voice) onset time
            if (opt=='VOwT')and(sig_audio is not None):
                VOwT.compute(sig_audio[ini_sample:end_sample],fs)
                res = VOwT.onset_label
                tm = get_segments(fs,res)[0]
                #Add the initial time of the segment that is being analyzed
                tm[0] = tm[0]+ini_time
                tm[1] = tm[1]+tm[0]
                Onsets['VOwT_'+str(f)] = np.hstack([tm[0],tm[1]]).reshape(1,-1)
        except:
            logging.info('Error computing Voice onset on a segment: Start [seconds]: '+str(np.round(ini_time,3))+' End [seconds]: '+str(np.round(end_time,3)))
        #Onset index
        f+=1
    return Onsets
#======================================================================
#======================================================================
#======================================================================        
        
def get_segments(fs,segments):
    """
    Get time stamps of the speech/voice segments after computing the energy contour

    Parameters
    ----------
    fs : Sampling frequency
    segments : array with labels of speech sounds

    Returns
    -------
    seg_time : List with time stamps

    """
    segments[0] = 0
    segments[-1:] = 0
    yp = segments.copy()
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    #Silence segments
    seg_time = []#Time stamps
    for idx in range(len(lim_ini)):
        #------------------------------------
        tini = lim_ini[idx]/fs
        tend = lim_end[idx]/fs
        seg_time.append([tini,tend])
        
    return seg_time