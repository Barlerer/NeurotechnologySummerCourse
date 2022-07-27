import numpy as np
from pylsl import StreamInlet, resolve_stream
import time
import pandas as pd
from mne_features.feature_extraction import FeatureExtractor
 

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


def save_stream(name: str) -> None:
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
    pd.DataFrame(np.hstack((timestamps, data))).to_csv(f'data/{name}.csv', index=False)
    print("Data saved")


def proccess_stream(data_path: str,sampling_frequency:int = 200) -> None:
    raw_data = pd.read_csv("data/test_run.csv")
    raw_data = raw_data.drop('0', axis=1).to_numpy().T
    data_lenght = raw_data.shape[1]
    window_size_in_sec = 2
    num_of_windows = round(data_lenght / (window_size_in_sec * sampling_frequency))
    windows = np.array_split(raw_data, num_of_windows,axis=1)
    data = np.stack(windows)
    fe = FeatureExtractor(sfreq=sampling_frequency, selected_funcs=['std'])


if __name__ == "__main__":
    # save_stream('test_run')
    proccess_stream('data/test_run.csv')
