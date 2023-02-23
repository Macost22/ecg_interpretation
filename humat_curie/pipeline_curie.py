import pandas as pd
import numpy as np
from ecg_feature_extraction.ecg_taxonomy import temporal_ecg_features
from ecg_feature_extraction.fiducial_point_detection import ecg_delineation, vector_fiducial

def load_data_arrhythmia(file_path):
    ecg = pd.read_csv(file_path, sep=" ", index_col=0)
    # Se transponen los datos  = (individuo, observaciones)
    ecg = ecg.transpose()
    # Se modifican los índices para que sean de 0 a n
    ecg.index = list(range(len(ecg)))
    return ecg


def main(signal, fs):
    ecg_fiducial_R, ecg_fiducial_points_nk, signal_filtered = ecg_delineation(signal=signal, fs=fs)
    signal_fiducial_zeros = np.zeros(signal.shape, dtype=int)
    vector_ecg_fiducial = vector_fiducial(ecg_fiducial_R, signal_fiducial_zeros)
    features = temporal_ecg_features(ecg_fiducial_R,signal_filtered,fs)
    return vector_ecg_fiducial,features


if __name__ == '__main__':
    # Estos tiene una fs= 250,  Wn_low = 60 y Wn_high = 0.5
    path_arritmia = 'C:/Users/melis/Desktop/Bioseñales/MIMIC/MIMIC_arritmia.txt'
    # signals = load_data_arrhythmia(path_arritmia)
    signal = load_data_arrhythmia(path_arritmia)

    fs = 250
    vector, features = main(signal.iloc[8], fs)

