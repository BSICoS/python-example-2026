"""EEG_processing.py

Este módulo contiene funciones para procesar datos EEG de los
hospitales incluidos en el desafío CincChallenge 2026. La principal
función definida es `MetricasHospitlal`, que recorre los archivos EDF
correspondientes a un hospital concreto, extrae las señales EEG,
las filtra, normaliza, crea épocas y calcula potencias de banda y
complejidades. Los resultados se guardan en un CSV resumen por
hospital.

Características principales:

- Soporta datos tanto del conjunto de entrenamiento como del
  conjunto suplementario.
- Selección automática de canales EEG a partir de la tabla
  `notebooks/channel_table.csv`.
- Creación de canales bipolares si están disponibles.
- Filtrado de banda 0.3-35 Hz y normalización de la señal.
- Re-muestreo a 200 Hz si fuese necesario.
- Cálculo de potencias de banda y complejidades usando
  funciones auxiliares (`lib/EEG_functions.py`).
- Exportación de resultados en `results_summaryEEG_{hospital}.csv`.

Uso típico:

>>> from src.scripts.EEG_processing import MetricasHospitlal
>>> MetricasHospitlal('I0002')

El módulo depende de `numpy`, `pandas`, `matplotlib`, `plotly` y de
las utilidades definidas en `lib/helper_code` y `lib/EEG_functions`.
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lib.helper_code as helper_code
import lib.EEG_functions as EEG_functions

def MetricasHospitlal(hospital):
    
    print(f"Procesando hospital: {hospital}")

    if hospital == 'I0002' or hospital == 'I0006' or hospital == "S0001":
        datapath =  'data/training_set/Physiological_data/'+hospital
    else:
        datapath =  'data/supplementary_set/Physiological_data/'+hospital
            
    channels = pd.read_csv("notebooks/channel_table.csv")
    selectEEG = channels[channels['Category'].isin(['eeg'])]

    demographics = pd.read_csv(os.path.join('C:/BSICoS/CincChallenge2026/CincChallenge_2026/data/training_set', "demographics.csv"))

    # Datos = pd.DataFrame(columns=['File', 'Channel', 'Sampling_Frequency', 'Duration_sec'])
    lista_dir = os.listdir(datapath)
    results = []

    for file in lista_dir:
        # Cargar el archivo (sustituye por tu ruta real)
        edf = helper_code.edfio.read_edf(os.path.join(datapath, file))

        id = file[9:-10]  # Asumiendo que el ID es el nombre del archivo sin la extensión
        
        selEEG = []
        labels = []
        data = []

        # Listar canales para identificar los de interés (ej: C3-M2, O1-M2)
        HayEEG = False
        for i, sig in enumerate(edf.signals):
            # print(f"[{i}] {sig.label}")
            # print length fs and duration
            # print(f"Length: {len(sig.data)}, Sampling Frequency: {sig.sampling_frequency} Hz, Duration: {len(sig.data)/sig.sampling_frequency:.2f} seconds")
            for index in selectEEG.index:
                if sig.label.lower() in selectEEG['Channel_Names'][index].lower():
                    print(f"Canal seleccionado: {sig.label}")
                    selEEG.append([i,sig])
                    labels.append(sig.label)
                    HayEEG = True
                    break
        # for i in range(len(edf.signals)):
        #     print(f"Longitud: {edf.signals[i].data.shape}, Canal: {edf.signals[i].label}, Frecuencia de muestreo: {edf.signals[i].sampling_frequency} Hz, Duración: {len(edf.signals[i].data)/edf.signals[i].sampling_frequency:.2f} segundos")
        
        if HayEEG:

            Bipolar = pd.DataFrame()
            if all(label in labels for label in ["F3", "F4", "M1", "M2"]):
                Bipolar['F3-M2'] = edf.signals[edf.labels.index("F3")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['F4-M1'] = edf.signals[edf.labels.index("F4")].data - edf.signals[edf.labels.index("M1")].data
                labels2 = ['F3-M2', 'F4-M1']
            if all(label in labels for label in ["C3", "C4", "M1", "M2"]):
                Bipolar['C3-M2'] = edf.signals[edf.labels.index("C3")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['C4-M1'] = edf.signals[edf.labels.index("C4")].data - edf.signals[edf.labels.index("M1")].data
                labels2.append('C3-M2')
                labels2.append('C4-M1')
            if all(label in labels for label in ["O2", "O1", "M1", "M2"]):
                Bipolar['O2-M2'] = edf.signals[edf.labels.index("O1")].data - edf.signals[edf.labels.index("M2")].data
                Bipolar['O1-M1'] = edf.signals[edf.labels.index("O2")].data - edf.signals[edf.labels.index("M1")].data
                labels2.append('O1-M1')
                labels2.append('O2-M2')
            # print(f"Archivo {file} tiene ECG, RESP y EEG. Se procesará con canales bipolares.")
            
            if not Bipolar.empty:
                labels = []
                for col in Bipolar.columns:
                    # print(f"Archivo: {file}, Canal: {col}, Frecuencia de muestreo: {sig.sampling_frequency} Hz, Duración: {len(Bipolar[col])/sig.sampling_frequency:.2f} segundos")
                    fs = edf.signals[edf.labels.index("M2")].sampling_frequency  # Asumimos que todos los canales tienen la misma frecuencia de muestreo
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
           
                
            demographics = demographics[demographics['BDSPPatientID'] == int(id)]
            print(demographics)
            
            for i, elec in enumerate(labels):
                epoch_length = 30  # Duración de cada época en segundos
                if Bipolar.empty:
                    fs = edf.signals[edf.labels.index(labels[i])].sampling_frequency
                else:
                    fs = edf.signals[edf.labels.index('M1')].sampling_frequency

                if fs != 200:
                    # print(f"Warning: Sampling frequency for channel {elec} in file {file} is {fs} Hz, expected 200 Hz. Check the data.")
                    duration = len(data[i]) / fs
                    time_original = np.linspace(0, duration, len(data[i]))
                    
                    num_samples_target = int(duration * 200 )
                    time_target = np.linspace(0, duration, num_samples_target)
                    data[i] = np.interp(time_target, time_original, data[i])
                    fs = 200  # Update fs to the target sampling frequency after resampling
                
                epochs = EEG_functions.create_epochs(data[i], fs, epoch_duration=epoch_length)

                band_powers, complexities = EEG_functions.extract_band_powers(epochs, fs, win_len=15)
                print(f"Band powers for file {file}:")
                
                band_powers = band_powers.iloc[60:]  # Eliminar las primeras 60 épocas (30 min) para evitar el tiempo despierto al inicio de la grabación
                print(band_powers.head())


                # # Convertir de formato "ancho" a "largo" para Plotly
                # df_melted = band_powers.melt(var_name='Banda', value_name='Potencia')
                # # Creamos el boxplot
                # fig = px.box(df_melted, x='Banda', y='Potencia', 
                #             color='Banda',
                #             points="outliers", # Para ver si hay épocas muy extrañas
                #             title=f"{id} - {elec} - {demographics.Cognitive_Impairment.values[0]}",
                #             log_y=True) # Usamos escala logarítmica porque Delta suele ser mucho más potente que Beta

                # fig.update_layout(template="plotly_white", showlegend=False)
                # # fig.write_html(f"graphs/BandasPersona/{id}_{elec}_{demographics.Cognitive_Impairment.values[0]}.html")  # Guardar como HTML para visualización interactiva
                # fig.show()

                # Ejecución
                patient_summar = EEG_functions.get_patient_profile(band_powers)
                # print(f"Resumen del perfil del paciente {id} - {elec}:")
                # print(patient_summar)
                d = complexities.iloc[:].std().to_dict() 
                results.append({
                    'File': file,
                    'Channel': elec,
                    'Patient_ID': id,
                    **d,
                    **patient_summar
                })
    df_results = pd.DataFrame(results)
    print(df_results.head())
    return df_results
    # df_results.to_csv(f"results_summaryEEG_{hospital}.csv", index=False)