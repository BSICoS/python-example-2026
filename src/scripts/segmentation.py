from binascii import Error
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import scipy.signal
from plotly.subplots import make_subplots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lib.helper_code as helper_code
import lib.EEG_functions as EEG_functions

for hospital in ['I0002','I0006', 'I0004','I0007','S0001']:
    print(f"Procesando hospital: {hospital}")

    if hospital == 'I0002' or hospital == 'I0006' or hospital == "S0001":
        datapath =  'data/training_set/Physiological_data/'+hospital
    else:
        datapath =  'data/supplementary_set/Physiological_data/'+hospital
        
    channels = pd.read_csv("notebooks/channel_table.csv")
    selectEEG = channels[channels['Category'].isin(['eeg'])]
    demographics = pd.read_csv(os.path.join('data/training_set', "demographics.csv"))
    selectresp = channels[channels['Category'].isin(['resp'])]
    selectECG = channels[channels['Category'].isin(['ecg'])]

    # Datos = pd.DataFrame(columns=['File', 'Channel', 'Sampling_Frequency', 'Duration_sec'])
    lista_dir = os.listdir(datapath)
    results = []

    for file in lista_dir:
        # Cargar el archivo (sustituye por tu ruta real)
        edf = helper_code.edfio.read_edf(os.path.join(datapath, file))

        id = file[9:-10]  # Asumiendo que el ID es el nombre del archivo sin la extensión
        selEEG = []
        selECG = []
        selResp = []
        labels = []
        data = []
        HayECG = False
        for i, sig in enumerate(edf.signals):
            for index in selectECG.index:
                if sig.label.lower() in selectECG['Channel_Names'][index].lower():
                    HayECG = True
                    print(f"Canal seleccionado: {sig.label}")
                    selECG.append([i,sig])
                    labels.append(sig.label)
                    data.append(sig.data)  # Guardar la señal ECG sin filtrar para su posterior procesamiento
                    break

        HayResp = False
        for i, sig in enumerate(edf.signals):
            for index in selectresp.index:
                if sig.label.lower() in selectresp['Channel_Names'][index].lower():
                    HayResp = True
                    fs = sig.sampling_frequency
                    if sig.label == "O2":
                        print(f"Warning: {sig.label} is detected as respiratory signal but has a sampling frequency higher than 100 Hz. Check the data.")
                    else:
                        print(f"Canal seleccionado: {sig.label}")
                        selResp.append([i,sig])
                        labels.append(sig.label)

                        data.append(sig.data)  # Guardar la señal RESP sin filtrar para su posterior procesamiento
                    break

        # Listar canales para identificar los de interés (ej: C3-M2, O1-M2)
        # print("Canales detectados:")

        HayEEG = False
        for i, sig in enumerate(edf.signals):
            # print(f"[{i}] {sig.label}")
            # print length fs and duration
            # print(f"Length: {len(sig.data)}, Sampling Frequency: {sig.sampling_frequency} Hz, Duration: {len(sig.data)/sig.sampling_frequency:.2f} seconds")
            for index in selectEEG.index:
                if sig.label.lower() in selectEEG['Channel_Names'][index].lower():
                    print(f"Canal seleccionado: {sig.label}")
                    selEEG.append([i,sig])
                    # labels.append(sig.label)
                    HayEEG = True
                    break
        # for i in range(len(edf.signals)):
        #     print(f"Longitud: {edf.signals[i].data.shape}, Canal: {edf.signals[i].label}, Frecuencia de muestreo: {edf.signals[i].sampling_frequency} Hz, Duración: {len(edf.signals[i].data)/edf.signals[i].sampling_frequency:.2f} segundos")
        
        if HayEEG and HayECG and HayResp:

            Bipolar = pd.DataFrame()
            if all(label in labels for label in ["F3", "F4", "M1", "M2"]):
                Bipolar['F3-M2'] = edf.signals[edf.labels.index("F3")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['F4-M1'] = edf.signals[edf.labels.index("F4")].data - edf.signals[edf.labels.index("M1")].data
            if all(label in labels for label in ["C3", "C4", "M1", "M2"]):
                Bipolar['C3-M2'] = edf.signals[edf.labels.index("C3")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['C4-M1'] = edf.signals[edf.labels.index("C4")].data - edf.signals[edf.labels.index("M1")].data
            if all(label in labels for label in ["O2", "O1", "M1", "M2"]):
                Bipolar['O2-M2'] = edf.signals[edf.labels.index("O1")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['O1-M1'] = edf.signals[edf.labels.index("O2")].data - edf.signals[edf.labels.index("M1")].data
            
            # print(f"Archivo {file} tiene ECG, RESP y EEG. Se procesará con canales bipolares.")
            if not Bipolar.empty:
                for col in Bipolar.columns:
                    # print(f"Archivo: {file}, Canal: {col}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(Bipolar[col])/sig.sampling_frequency:.2f} segundos")
                    fs = edf.signals[edf.labels.index("O2")].sampling_frequency  # Asumimos que todos los canales tienen la misma frecuencia de muestreo
                    time = np.linspace(0, len(Bipolar[col]) / fs, len(Bipolar[col]))
                    fil = EEG_functions.butter_bandpass_filter(Bipolar[col], lowcut=0.3, highcut=35, fs=fs, order=4)
                    norm = (fil-np.mean(fil))/np.std(fil)
                    
                    data.append(norm)  # Restar la media para centrar la señal
                    labels.append(col)
                # columns = Bipolar.columns.tolist()
            else:
                for i, (idx, sig) in enumerate(selEEG):
                    # print(f"Archivo: {file}, Canal: {sig.label}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(sig.data)/sig.sampling_frequency:.2f} segundos")
                    fs = sig.sampling_frequency
                    time = np.linspace(0, len(sig.data) / fs, len(sig.data))
                    fil = EEG_functions.butter_bandpass_filter(sig.data, lowcut=0.3, highcut=35, fs=fs, order=4)
                    norm = (fil-np.mean(fil))/np.std(fil)
                    labels.append(sig.label)
                    data.append(norm)  # Restar la media para centrar la señal

                # columns = [selEEG[i][1].label for i in range(len(selEEG))]


            # for i in range(len(selResp)):
                # columns.append(selResp[i][1].label)
            demographicsID = demographics[demographics['BDSPPatientID'] == int(id)]
            print(demographicsID)

            columnashoras = []
            for elec in labels:
                for h in np.floor(np.arange(0, len(sig.data) / fs / 3600, 1)):
                    columnashoras.append(elec + f"_h{int(h)}")
            epochs5min = pd.DataFrame(columns= columnashoras)
                
            for i, elec in enumerate(labels):
                # Check fs of the current channel
                if elec == 'O2_resp':
                    fs = edf.signals[edf.labels.index('O2')].sampling_frequency
                else:
                    fs = edf.signals[edf.labels.index(labels[i])].sampling_frequency

                if fs != 200:
                    # print(f"Warning: Sampling frequency for channel {elec} in file {file} is {fs} Hz, expected 200 Hz. Check the data.")
                    duration = len(data[i]) / fs
                    time_original = np.linspace(0, duration, len(data[i]))
                    
                    num_samples_target = int(duration * 200 )
                    time_target = np.linspace(0, duration, num_samples_target)
                    data[i] = np.interp(time_target, time_original, data[i])
                    fs = 200  # Update fs to the target sampling frequency after resampling

                    # Plot comparison of original and resampled signals
                    # lim = 50000
                    # factor = len(filtered_data[0]) / num_samples_target
                    # plt.figure(figsize=(12, 6))
                    # plt.plot(time_target[:int(lim/factor)], resampled_data[:int(lim/factor)], label='Resampled Signal')
                    # plt.plot(time_original[:lim], filtered_data[i][:lim], label='Original Signal')
                    # plt.title(f'Original vs Resampled Signal - {elec} in {file}')
                    # plt.show()


                epoch_length = 300  # Duración de cada época en segundos
                # epochs = EEG_functions.create_epochs(df[elec].values, fs, epoch_duration=epoch_length)
                epochs = EEG_functions.create_epochs(data[i], fs, epoch_duration=epoch_length)

                # Coger los primeros 5min de cada hora
                for h in np.floor(np.arange(0, len(epochs)*epoch_length/3600, 1)):
                    start_epoch = int(h*3600/epoch_length)
                    end_epoch = int((h*3600 + 5*60)/epoch_length)
                    if end_epoch > len(epochs):
                        end_epoch = len(epochs)
                    c = elec+'_h'+str(int(h))
                    epochs5min.loc[:, c] = epochs[start_epoch:end_epoch][0]
                    
                
                # del epochs, fs   # Liberar memoria

            # Plotly sublot epochs5min.iloc[:,::8].plot()
            # h = 6
            # fig = make_subplots(rows=int(epochs5min.shape[1]/8), cols=1, subplot_titles=epochs5min.columns[h::8])
            # for i in range(h, epochs5min.shape[1], 8):
            #     print(f"Plotting channel: {epochs5min.columns[i]}")
            #     fig.add_trace(go.Scatter(x=epochs5min.index, y=epochs5min.iloc[:,i], mode='lines', name=epochs5min.columns[i]), row=int(i/8)+1, col=1)
            # fig.update_layout(height=3000, width=1200, title_text=f"Epochs de 5 minutos para el archivo {file}")
            # fig.show()
            if epochs5min.shape[1] > 0 and epochs5min.shape[0] == 60000:
                epochs5min.to_parquet(os.path.join('X:/bsicos01/__comun/Physionet/Data5min', f"{id}.parquet"))
            else:
                print(f"Error: No channels in epochs5min for file {file}")
                # stop program if no channels are processed
                raise ValueError(f"No channels in epochs5min for file {file}")