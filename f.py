
import os
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def convert_mp3_to_wav_tensorflow(mp3_folder_path, wav_folder_path):
    # Create the output folder if it does not exist
    os.makedirs(wav_folder_path, exist_ok=True)

    # Loop through all MP3 files in the input folder
    for mp3_file_path in os.listdir(mp3_folder_path):
        if mp3_file_path.endswith(".mp3"):
            mp3_file_path = os.path.join(mp3_folder_path, mp3_file_path)

            
            audio, _ = librosa.load(mp3_file_path, sr=None)

            wav_file_path = os.path.join(
                wav_folder_path, os.path.splitext(os.path.basename(mp3_file_path))[0] + ".wav"
            )

            librosa.output.write_wav(wav_file_path, audio, sr=44100)

    print(f"All MP3 files in {mp3_folder_path} have been converted to WAV format using TensorFlow and saved in {wav_folder_path}")

if __name__ == "__main__":
