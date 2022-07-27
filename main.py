import numpy as np
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
data = []
timestamps = []
sample_points = 200
for _ in range(sample_points):
    sample, timestamp = inlet.pull_sample()
    data.append(sample)
    timestamps.append(timestamp)


data = np.array(data)
timestamps = np.array(timestamps)


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


freq, PSD = power_spectrum(data, timestamps)
