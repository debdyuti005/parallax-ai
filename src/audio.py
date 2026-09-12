import sounddevice as sd
import numpy as np

# Placeholders and global variables
SOUND_AMPLITUDE = 0
AUDIO_CHEAT = 0

# Sound variables
CALLBACKS_PER_SECOND = 38               # Callbacks per second (system dependent)
SUS_FINDING_FREQUENCY = 2               # Calculates SUS *n* times every second
SOUND_AMPLITUDE_THRESHOLD = 0.8        # Amplitude considered for SUS calc (adjusted for normal mic levels) 

# Packing *n* frames to calculate SUS
FRAMES_COUNT = int(CALLBACKS_PER_SECOND / SUS_FINDING_FREQUENCY)
AMPLITUDE_LIST = [0] * FRAMES_COUNT
SUS_COUNT = 0
count = 0

def calculate_rms(indata):
    """Calculate the Root Mean Square (RMS) value of the audio data."""
    return np.sqrt(np.mean(indata**2)) * 1000  # Scaling factor for better readability

def print_sound(indata, outdata, frames, time_info, status):
    global SOUND_AMPLITUDE, SUS_COUNT, count, SOUND_AMPLITUDE_THRESHOLD, AUDIO_CHEAT
    rms_amplitude = calculate_rms(indata)
    AMPLITUDE_LIST.append(rms_amplitude)
    count += 1
    AMPLITUDE_LIST.pop(0)
    
    if count == FRAMES_COUNT:
        avg_amp = sum(AMPLITUDE_LIST) / FRAMES_COUNT
        SOUND_AMPLITUDE = avg_amp
        
        if SUS_COUNT >= 2:
            AUDIO_CHEAT = 1
            SUS_COUNT = 0
            print(f"AUDIO CHEAT DETECTED! Amplitude: {avg_amp:.2f}")
        
        if avg_amp > SOUND_AMPLITUDE_THRESHOLD:
            SUS_COUNT += 1
            print(f"Audio above threshold: {avg_amp:.2f} > {SOUND_AMPLITUDE_THRESHOLD}")
        else:
            SUS_COUNT = 0
            AUDIO_CHEAT = 0
        
        count = 0

def sound():
    try:
        print("Starting audio detection...")
        with sd.Stream(callback=print_sound, channels=1, samplerate=44100):
            sd.sleep(-1)
    except Exception as e:
        print(f"Audio detection error: {e}")
        print("Available audio devices:")
        print(sd.query_devices())

def sound_analysis():
    global AMPLITUDE_LIST, FRAMES_COUNT, SOUND_AMPLITUDE
    while True:
        AMPLITUDE_LIST.append(SOUND_AMPLITUDE)
        AMPLITUDE_LIST.pop(0)

        avg_amp = sum(AMPLITUDE_LIST) / FRAMES_COUNT

        if avg_amp > 10:
            print("Sus...")

if __name__ == "__main__":
    sound()
