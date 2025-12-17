import numpy as np
import sounddevice as sd

fs = 16000  # Sample rate
seconds = 3  # Duration of recording

print("Testing recording for 3 seconds...")
try:
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    print(f"Max amplitude: {np.max(np.abs(myrecording))}")
except Exception as e:
    print(f"Error: {e}")
