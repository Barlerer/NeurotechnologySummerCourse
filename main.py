import os
import time

import numpy as np
import pandas as pd
from mne_features.feature_extraction import FeatureExtractor
from pylsl import StreamInlet, resolve_stream
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SAMPLING_FREQUENCY = 200
LABEL = {'l':0,'r':1,'i':2}
def power_spectrum(signal, timestamps):
    dt = np.mean(np.diff(timestamps))
    window_size = timestamps[-1] - timestamps[0]

    n = signal.shape[0]
    fft_signal = np.fft.fft(signal, axis=0)
    PSD = np.abs(fft_signal)
    window_size = signal.shape[0] * dt
    t = np.arange(0, window_size, dt)
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(n // 2)
    return freq[L], PSD[L] / n


def save_stream(name: str,label:str) -> None:
    """Saves the stream to CSV file

    Parameters
    ----------
    name : str
        Name of the file
    label : str
        Should be either l,r,i
    """
    # Parameter to define the experiment time, in seconds
    recording_length = 120
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    data = []
    timestamps = []
    start_time = time.time()
    while time.time() - start_time <= recording_length:
        sample, timestamp = inlet.pull_sample()
        data.append(sample)
        timestamps.append(timestamp)
    data = np.array(data)
    timestamps = np.expand_dims(np.array(timestamps), axis=-1)
    # Save the data to csv
    pd.DataFrame(np.hstack((timestamps, data))).to_csv(f'data/{name}_{label}.csv', index=False)
    print("Data saved")


def proccess_stream(data_path: str,sampling_frequency:int = 200) -> np.ndarray:
    window_size_in_sec = 3

    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.drop(['0','1','2'], axis=1).to_numpy().T
    # This is how many sample we got
    data_len = raw_data.shape[1]
    num_of_windows = round(data_len / (window_size_in_sec * sampling_frequency))
    # We want to remove some data so that we would fit into equal windows
    raw_data=raw_data[:,:data_len - (data_len % num_of_windows)]
    windows = np.array_split(raw_data, num_of_windows,axis=1)
    return np.stack(windows)

def extract_features():
    params = {'pow_freq_bands__freq_bands': np.array([[8.,12.],[19.,23.]])}
    return FeatureExtractor(sfreq=SAMPLING_FREQUENCY, selected_funcs=['pow_freq_bands','std'],params=params)

if __name__ == "__main__":
    # save_stream('test_run')
    # assign directory
    directory = 'data'
    
    # iterate over files in
    # that directory
    X = []
    Y = []
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)
        data=proccess_stream(file,SAMPLING_FREQUENCY)
        fe = extract_features()
        extracted_features=fe.fit_transform(data)

        # Set up a label for our data
        labels=np.ones(extracted_features.shape[0]) * LABEL[file[-5:-4]]
        X.append(extracted_features)
        Y.append(labels)
    X = np.concatenate(X)
    Y = np.concatenate(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    classfier=SVC(probability=True)
    classfier.fit(X_train,y_train)
    classification_accuracy=classfier.score(X_test,y_test)
    print(classification_accuracy)

