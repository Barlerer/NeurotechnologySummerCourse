import time
from pylsl import StreamInlet, resolve_stream
import numpy as np
import pandas as pd
import argparse

def main(name: str,label:str) -> None:
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Saves LSL Stream to CSV file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", help="name of file")
    parser.add_argument("-l", "--label", help="label of recording, should be either 'l', 'r', 'i'")

    args = parser.parse_args()
    config = vars(args)
    main(config.get('name',''),config.get('label',''))