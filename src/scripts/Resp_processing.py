import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import sys 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lib.helper_code as helper_code
import lib.EEG_functions as EEG_functions
import lib.Resp_features as Resp_features

for hospital in ['I0006',"S0001",'I0004','I0007']:#'I0002',
    print(f"Procesando hospital: {hospital}")

    if hospital == 'I0002' or hospital == 'I0006' or hospital == "S0001":
        datapath =  'data/training_set/Physiological_data/'+hospital
    else:
        datapath =  'data/supplementary_set/Physiological_data/'+hospital
            
    channels = pd.read_csv("notebooks/channel_table.csv")
    selectResp = channels[channels['Category'].isin(['resp'])]

    demographics = pd.read_csv(os.path.join('C:/BSICoS/CincChallenge2026/CincChallenge_2026/data/training_set', "demographics.csv"))

    # Datos = pd.DataFrame(columns=['File', 'Channel', 'Sampling_Frequency', 'Duration_sec'])
    lista_dir = os.listdir(datapath)
    results = []

    for file in lista_dir:
        # Cargar el archivo (sustituye por tu ruta real)
        edf = helper_code.edfio.read_edf(os.path.join(datapath, file))

        id = file[9:-10]  # Asumiendo que el ID es el nombre del archivo sin la extensión
        
        selResp = []
        labels = []
        data = []

        HayResp = False
        for i, sig in enumerate(edf.signals):
            for index in selectResp.index:
                if sig.label.lower() in selectResp['Channel_Names'][index].lower():
                    print(f"Canal seleccionado: {sig.label}")
                    selResp.append([i,sig])
                    labels.append(sig.label)
                    HayResp = True
                    # plot en plotly la señal
                    go.Figure(data=go.Scattergl(x=np.arange(len(sig.data))/sig.sampling_frequency, y=sig.data, mode='lines', name=sig.label)).update_layout(title=f"Señal de {sig.label} - Archivo: {file}", xaxis_title="Tiempo (s)", yaxis_title="Amplitud").show()
                    # px.line(x=np.arange(len(sig.data))/sig.sampling_frequency, y=sig.data, title=f"Señal de {sig.label} - Archivo: {file}").show()
                    break
        
        if HayResp:
            for i, (idx, sig) in enumerate(selResp):
                print(f"Archivo: {file}, Canal: {sig.label}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(sig.data)/sig.sampling_frequency:.2f} segundos")
                fs = sig.sampling_frequency

                if fs != 25:
                    duration = len(sig.data) / fs
                    time_original = np.linspace(0, duration, len(sig.data))
                    num_samples_target = int(duration * 25 )
                    time_target = np.linspace(0, duration, num_samples_target)
                    data = np.interp(time_target, time_original, sig.data)
                    fs = 25  # Update fs to the target sampling frequency after resampling
                else:
                    data = sig.data
                    time_new = np.linspace(0, len(sig.data) / fs, len(sig.data))

                # Check nan in sig.data
                if np.isnan(sig.data).any():
                    print(f"Warning: NaN values found in signal data for {sig.label}. Filling NaNs with zeros.")
                    data = np.nan_to_num(data)

                # if sig.label not in ["SpO2", "SaO2", "OSAT", "O2SAT", "O2 SAT", "O2-SAT", "O2-SATURATION"]:
                #     fil = EEG_functions.butter_bandpass_filter(data, lowcut=0.01, highcut=4, fs=fs, order=4)
                #     # norm = (fil-np.mean(fil))/np.std(fil)
                #     data.append(fil)  # Restar la media para centrar la señal
                
                if sig.label.lower() in selectResp['Channel_Names'][28].lower() or sig.label.lower() in selectResp['Channel_Names'][29].lower():
                    # EFFORT RESPIRATORY
                elif sig.label.lower() in selectResp['Channel_Names'][30].lower() or sig.label.lower() in selectResp['Channel_Names'][31].lower():
                    # RESPIRATORY Flujo
                    fil = EEG_functions.butter_bandpass_filter(data, lowcut=0.01, highcut=4, fs=fs, order=4)
                    Resp_features.peakedness_application(fil, stage=sig.label, plotflag = True, subjet =1) 
                elif sig.label.lower() in selectResp['Channel_Names'][32].lower() or sig.label.lower() in selectResp['Channel_Names'][33].lower():
                    # CEPAP
                elif sig.label.lower() in selectResp['Channel_Names'][34].lower():
                    #O2 SATURATION



                # time_dt = pd.to_datetime(time_new, unit='s')    
                # # Plot raw and filtered signals
                # fig = make_subplots(specs=[[{"secondary_y": True}]])
                # fig.add_trace(go.Scattergl(x=time_dt[::10], y=data[::10], name=sig.label, mode='lines'),secondary_y=False,row=1, col=1)
                # fig.add_trace(go.Scattergl(x=time_dt[::10], y=fil[::10], name=f"Normalized {sig.label}", mode='lines'), secondary_y=True,row=1, col=1)
                # fig.update_yaxes(title_text="Amplitud Original (uV)", secondary_y=False)
                # fig.update_yaxes(title_text="Valor Normalizado (Z-score)", secondary_y=True)
                # # update x axis to make time format
                # fig.update_xaxes(
                #     tickformat="%H:%M:%S", # Formato de hora:minuto:segundo
                #     row=1, col=1
                # )
                # fig.show()

                #  Plot spectrogram of raw and filtered signals
                # fig = make_subplots(specs=[[{"secondary_y": True}]])
                # fig.add_trace(go.Scattergl(x=time_dt[::10], y=data[::10], name=sig.label, mode='lines'),secondary_y=False,row=1, col=1)
                # fig.add_trace(go.Scattergl(x=time_dt[::10], y=fil[::10], name=f"Normalized {sig.label}", mode='lines'), secondary_y=True,row=1, col=1)
                # fig.update_yaxes(title_text="Amplitud Original (uV)", secondary_y=False)

