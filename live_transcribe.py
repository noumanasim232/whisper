import queue
import sys
import threading
import time

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd

import whisper

# Configuration
MODEL_TYPE = "small"  # 'small' is much better for non-English languages like Hindi
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024  # Audio chunk size
SILENCE_THRESHOLD_MULTIPLIER = (
    2.0  # Multiplier for ambient noise to determine silence threshold
)
SILENCE_DURATION = 2.0  # Seconds of silence to mark end of speech
MIN_RECORDING_DURATION = 0.5  # Minimum seconds of speech to transcribe

# Queue for communication between audio thread and transcription thread
transcription_queue = queue.Queue()


def calculate_rms(audio_chunk):
    """Calculate Root Mean Square amplitude of the audio chunk."""
    return np.sqrt(np.mean(audio_chunk**2))


def calibrate_noise(duration=2.0):
    """Calibrate the silence threshold based on ambient noise."""
    print(f"Calibrating ambient noise for {duration} seconds... Please remain silent.")

    # Use blocking recording instead of callback to avoid potential hangs
    recording = sd.rec(
        int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS
    )
    sd.wait()

    # Calculate RMS for chunks to simulate the block processing
    # Reshape into blocks to match the main loop's processing
    num_blocks = len(recording) // BLOCK_SIZE
    # Truncate to full blocks
    recording = recording[: num_blocks * BLOCK_SIZE]
    blocks = np.split(recording, num_blocks)

    rms_values = [calculate_rms(chunk) for chunk in blocks]
    avg_rms = np.mean(rms_values)
    max_rms = np.max(rms_values)

    threshold = max_rms * SILENCE_THRESHOLD_MULTIPLIER

    # Ensure a minimum threshold to avoid triggering on pure silence/noise floor
    min_threshold = 0.001
    threshold = max(threshold, min_threshold)

    print(
        f"Calibration complete. Average RMS: {avg_rms:.5f}, Max RMS: {max_rms:.5f}, Threshold: {threshold:.5f}"
    )
    return threshold


def transcribe_worker(model):
    """Worker function to process audio from the queue."""
    while True:
        audio_data = transcription_queue.get()
        if audio_data is None:
            break

        try:
            # Whisper expects a flattened float32 array
            audio_flattened = audio_data.flatten().astype(np.float32)

            duration = len(audio_flattened) / SAMPLE_RATE
            print(f"Transcribing {duration:.2f} seconds of audio...")

            # Transcribe
            result = model.transcribe(audio_flattened, fp16=False)
            text = result["text"].strip()

            if text:
                print(f"Transcribed: '{text}'")
                type_text(text)
            else:
                print("No text transcribed.")

        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            transcription_queue.task_done()


def type_text(text):
    """Type the text using pyautogui."""
    if not text:
        return
    print(f"Typing: {text}")
    # Copy to clipboard and paste is often faster and safer for special chars
    pyperclip.copy(text + " ")
    # On Mac, use Command+V
    pyautogui.hotkey("command", "v")


def main():
    print(f"Loading Whisper model '{MODEL_TYPE}'...")
    model = whisper.load_model(MODEL_TYPE)
    print("Model loaded.")

    # Start transcription worker thread
    worker_thread = threading.Thread(
        target=transcribe_worker, args=(model,), daemon=True
    )
    worker_thread.start()

    threshold = calibrate_noise()

    print("Listening... (Press Ctrl+C to stop)")

    audio_buffer = []
    is_recording = False
    silence_start_time = None

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK_SIZE
        ) as stream:
            while True:
                indata, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    # This might still happen if the system is very heavily loaded, but much less likely now
                    print("Audio buffer overflowed (input)", file=sys.stderr)

                rms = calculate_rms(indata)

                if rms > threshold:
                    # Speech detected
                    if not is_recording:
                        print("Speech detected, starting recording...")
                        is_recording = True
                        audio_buffer = []  # Reset buffer

                    audio_buffer.append(indata.copy())
                    silence_start_time = None  # Reset silence timer

                elif is_recording:
                    # Silence detected while recording
                    audio_buffer.append(indata.copy())

                    if silence_start_time is None:
                        silence_start_time = time.time()

                    # Check if silence has exceeded duration
                    if time.time() - silence_start_time > SILENCE_DURATION:
                        print("End of speech detected.")

                        # Process the recording
                        full_audio = np.concatenate(audio_buffer, axis=0)
                        duration = len(full_audio) / SAMPLE_RATE

                        if duration >= MIN_RECORDING_DURATION:
                            # Offload to worker thread
                            transcription_queue.put(full_audio)
                        else:
                            print("Recording too short, discarding.")

                        # Reset state
                        is_recording = False
                        audio_buffer = []
                        silence_start_time = None
                        print("Listening...")

    except KeyboardInterrupt:
        print("\nStopping...")
        transcription_queue.put(None)  # Signal worker to stop
        worker_thread.join()


if __name__ == "__main__":
    main()
