# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 22:58:50 2023

@author: Melissa
"""

import pandas as pd
from normalized_fiducial_points import *
from fiducial_point_detection import ecg_delineation


normalized_fiducial_personality_traits= []


def load_data_personality_traits(file_path):
    ecg_70 = pd.read_csv(file_path, sep=" ")
    # Se transponen los datos  (68,240000) = (individuo, observaciones)
    ecg_70 = ecg_70.transpose()
    # Se modifican los índices para que sean de 0 a 67
    ecg_70.index = list(range(len(ecg_70)))
    # ecg1=ecg_70.iloc[1]
    return ecg_70


def load_data_arrhythmia(file_path):
    ecg = pd.read_csv(file_path, sep=" ", index_col=0)
    # Se transponen los datos  (68,240000) = (individuo, observaciones)
    ecg = ecg.transpose()
    # Se modifican los índices para que sean de 0 a 67
    ecg.index = list(range(len(ecg)))
    # ecg1=ecg_70.iloc[1]
    return ecg


def ecg_features(path, fs):
    signals = load_data_personality_traits(path)
    # Se analiza cada ecg dentro del ciclo for, para así obtener los puntos fiduciales
    for signal in range(len(signals)):

        print('persona: {}'.format(signal))

        ecg = signals.iloc[signal]
        ecg_fiducial_R, ecg_fiducial_points_nk = ecg_delineation(signal=ecg, fs=fs)

        normalized_fiducial_personality_traits.append(normalized_fiducial(ecg_fiducial_R, fs))
    flatten_normalized_personality_traits = flatten_values(normalized_fiducial_personality_traits)

    return flatten_normalized_personality_traits


if __name__ == '__main__':
    # Path de los ecg a analizar
    # Estos tiene una fs= 2000,  Wn_low = 100 y Wn_high = 1
    path='C:/Users/melis/Desktop/Bioseñales/ECG_veronica/ecg_70.txt'

    fs = 2000
    flatten_normalized_pt = ecg_features(path,fs)
    fiducial_df = pd.DataFrame.from_dict(flatten_normalized_pt)
    fiducial_df.to_csv('features_ecg_personality_traits.csv')





 


   
            
            
 
