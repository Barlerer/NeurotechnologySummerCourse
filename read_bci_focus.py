from time import time
from httpcore import stream
from pylsl import StreamInlet, resolve_stream
import httpx

API_URL = 'https://slothy1.herokuapp.com/postCommand?command='
ACTIONS = {0:'0',1:'1'}
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
SECONDS = 3

while True:
    start = time()
    stream_window = []
    while time() - start < SECONDS:
        sample, timestamp = inlet.pull_sample()
        stream_window.append(sample[0])
    majortiy_action = max(stream_window,key=stream_window.count)
    # 0 is non relax, 1 is
    # Now we can send it as a command to the API
    print(API_URL + ACTIONS[majortiy_action])

    # Uncomment the line below for the demo
    # httpx.get(API_URL + ACTIONS[majortiy_action])