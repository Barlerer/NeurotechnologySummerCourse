import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
import pandas as pd
from mne_features.feature_extraction import FeatureExtractor
from sklearn.svm import SVC

SAMPLING_FREQUENCY = 200
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
    window_size_in_sec = 2

    raw_data = pd.read_csv("data/test_run.csv")
    raw_data = raw_data.drop('0', axis=1).to_numpy().T
    # This is how many sample we got
    data_len = raw_data.shape[1]
    num_of_windows = round(data_len / (window_size_in_sec * sampling_frequency))
    # We want to remove some data so that we would fit into equal windows
    raw_data=raw_data[:,:data_len - (data_len % num_of_windows)]
    windows = np.array_split(raw_data, num_of_windows,axis=1)
    return np.stack(windows)

def extract_features():
    params = {'pow_freq_bands__freq_bands': np.array([[8.,12.],[15.,25.]])}
    return FeatureExtractor(sfreq=SAMPLING_FREQUENCY, selected_funcs=['pow_freq_bands','std'],params=params)

if __name__ == "__main__":
    save_stream('test_run')
    # data=proccess_stream('data/test_runs.csv',SAMPLING_FREQUENCY)
    # fe = extract_features()
    # extracted_features=fe.fit_transform(data)

    # ###
    # # Just for example, later you need to create a proper label
    # ###
    # label = np.zeros(extracted_features.shape[0])
    # label[0] = 1
    # label[1] = 2

    # classfier=SVC()
    # classfier.fit(extracted_features,label)
    # classification_accuracy=classfier.score(extracted_features,label)
    # print(classification_accuracy)

