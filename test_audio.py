import src.audio as audio
import threading
import time

print("Starting audio test...")
print(f"Initial values - SOUND_AMPLITUDE: {audio.SOUND_AMPLITUDE}, AUDIO_CHEAT: {audio.AUDIO_CHEAT}")

# Start audio detection in a separate thread
audio_thread = threading.Thread(target=audio.sound, daemon=True)
audio_thread.start()

# Wait and check values
for i in range(10):
    time.sleep(1)
    print(f"After {i+1}s - SOUND_AMPLITUDE: {audio.SOUND_AMPLITUDE:.2f}, AUDIO_CHEAT: {audio.AUDIO_CHEAT}")

print("Test completed!")