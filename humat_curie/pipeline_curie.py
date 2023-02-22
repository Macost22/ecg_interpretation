# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:44:22 2022

@author: Melissa
"""
import pandas as pd
from fiducial_point_detection import ecg_delineation
from normalized_fiducial_points import *
import neurokit2 as nk

normalized_fiducial_arrhythmia= []


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


def main(path, fs):
    signals = load_data_arrhythmia(path)
        # Se analiza cada ecg dentro del ciclo for, para así obtener los puntos fiduciales
    for signal in range(len(signals)):

        print('paciente: {}'.format(signal))

        ecg = signals.iloc[signal]
        ecg_fiducial_R, ecg_fiducial_points_nk = ecg_delineation(signal=ecg, fs=fs)

        normalized_fiducial_arrhythmia.append(normalized_fiducial(ecg_fiducial_R, fs))
    flatten_normalized_fiducial_arritmia = flatten_values(normalized_fiducial)

    return flatten_normalized_fiducial_arritmia


if __name__ == '__main__':

    # Estos tiene una fs= 250,  Wn_low = 60 y Wn_high = 0.5
    # path_arritmia = 'C:/Users/melis/Desktop/Bioseñales/MIMIC/MIMIC_arritmia.txt'
    path='C:/Users/melis/Desktop/Bioseñales/ECG_veronica/ecg_70.txt'
    # signals = load_data_arrhythmia(path_arritmia)
    
    fs = 2000
    fv = main(path, fs)





    
    # Se analiza cada ecg dentro del ciclo for, para así obtener los puntos fiduciales
    #for signal in range(len(signals)):
    #        print('paciente: {}'.format(signal))
    #        ecg = signals.iloc[signal]
    #        ecg_fiducial, ecg_fiducial_nk = main(signal = ecg, fs = 250, Wn_low = 60, Wn_high = 0.5)
    #
    #        ecg_fiducial['signal']=ecg
    #        ecg_fiducial_nk['signal']=ecg
            
     #       fiducial.append(ecg_fiducial)
     #       fiducial_nk.append(ecg_fiducial_nk)
           
    # for persona in range(len(fiducial)):
    #      paciente = fiducial[persona]
    #      paciente_nk = fiducial_nk[persona]
    #      signal = fiducial[persona]['signal']       
         
      
    #      print('Paciente {} \n'.format(persona))
    #      print('Taxonomia con R')
    #      taxonomy(paciente, signal,fs)
         
    #      print('-----------------------------')
    #      print('Taxonomía con neurokit2 \n')
    #      taxonomy(paciente_nk, signal,fs)
    #      print('###################################')

                
