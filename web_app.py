import csv
import time
import json
import sys
import threading
import web
import nia as NIA
import serial
from urllib.parse import unquote, quote
import os
import signal
import numpy as np
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

urls = (
    '/', 'index',
    '/get_steps', 'get_steps',
    '/shutdown', 'shutdown'
)

# global scope stuff
nia = None
nia_data = None
running = True  # Indicateur pour contrôler l'exécution des threads

try:
    serial_port = serial.Serial('COM5', 921600)
    print("Port série COM5 ouvert avec succès")
except serial.SerialException as e:
    print(f"Erreur: Impossible d'ouvrir le port série COM5: {e}")

class index:
    def GET(self):
        render = web.template.render("templates/")
        return render.index()

class get_steps:
    def GET(self):
        web.header("Content-Type", "application/json")
        data = {
            "brain_fingers": web.brain_fingers
        }
        return json.dumps(data)

class shutdown:
    def GET(self):
        global running
        running = False  # Arrêter l'exécution des threads
        threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGINT)).start()

class CSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = ['timestamp', 'eeg_pure', 'low_alpha', 'med_alpha', 'high_alpha', 'low_beta', 'med_beta', 'high_beta','delta', 'theta', 'alpha','beta', 'brain_state']
        self.write_header()

    def write_header(self):
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_row(self, data):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(data)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Fréquence de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0 or high >= 1:
        raise ValueError("Les fréquences critiques doivent être dans l'intervalle (0, 1).")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def calculate_amplitudes(eeg_data, fs):
    delta = bandpass_filter(eeg_data, 0.5, 3.9, fs)
    theta = bandpass_filter(eeg_data, 4.0, 7.9, fs)
    alpha = bandpass_filter(eeg_data, 8.0, 11.9, fs)
    beta = bandpass_filter(eeg_data, 12.0, 19.9, fs)

    delta_amp = np.mean(np.abs(delta))
    theta_amp = np.mean(np.abs(theta))
    alpha_amp = np.mean(np.abs(alpha))
    beta_amp = np.mean(np.abs(beta))

    return delta_amp, theta_amp, alpha_amp, beta_amp

def determine_brain_state(delta, theta, alpha, beta):
    if delta > theta and delta > alpha and delta > beta:
        return "Relaxation", "blue"
    elif theta > delta and theta > alpha and theta > beta:
        return "Somnolence", "green"
    elif alpha > delta and alpha > theta and alpha > beta:
        return "Calme", "yellow"
    elif beta > delta and beta > theta and beta > alpha:
        return "Concentration", "red"
    else:
        return "Neutre", "white"

def calculate_spectrogram(eeg_data, fs):
    f, t, Sxx = spectrogram(eeg_data, fs, nperseg=min(len(eeg_data), 128))
    Sxx_log = 10 * np.log10(Sxx)  # Convertir en échelle logarithmique

    norm = plt.Normalize(vmin=np.min(Sxx_log), vmax=np.max(Sxx_log))
    cmap = plt.get_cmap('viridis')
    colors_rgb = cmap(norm(Sxx_log))
    colors_rgb = (colors_rgb[:, :, :3] * 255).astype(np.uint8)  # Convertir en valeurs RGB

    return colors_rgb

def send_spectrogram_to_arduino(colors_rgb):
    # Envoyer les données du spectrogramme au port série
    height, width, _ = colors_rgb.shape
    for j in range(height):
        for i in range(width):
            r, g, b = colors_rgb[j, i]
            color_string = bytes([r, g, b])
            serial_port.write(color_string)
            time.sleep(0.001)  # Ajuster la vitesse d'envoi si nécessaire
    print("Spectrogram data sent")

class Updater:
    def __init__(self, csv_writer):
        self.csv_writer = csv_writer

    def update(self):
        global running
        while running:
            # kick-off processing data from the NIA
            data_thread = threading.Thread(target=nia_data.get_data)
            data_thread.start()

            # get the fourier data from the NIA
            data, steps = nia_data.fourier(nia_data)
            web.brain_fingers = steps

            # wait for the next batch of data to come in
            data_thread.join()

            # Prepare the data to write into CSV
            eeg_data = nia_data.Raw_Data
            timestamp = time.time()

            # Calculate the amplitude of the different frequency bands
            delta_amp, theta_amp, alpha_amp, beta_amp = calculate_amplitudes(eeg_data, 40)

            # Determine the brain state
            brain_state, state_color = determine_brain_state(delta_amp, theta_amp, alpha_amp, beta_amp)

            # Create the row for the CSV
            row = {
                'timestamp': time.time(),
                'eeg_pure': eeg_data.tolist(),
                'low_alpha': steps[0],
                'med_alpha': steps[1],
                'high_alpha': steps[2],
                'low_beta': steps[3],
                'med_beta': steps[4],
                'high_beta': steps[5],
                'delta': delta_amp,
                'theta': theta_amp,
                'alpha': alpha_amp,
                'beta': beta_amp,
                'brain_state' : brain_state
            }
            self.csv_writer.write_row(row)

            # Calculate and send spectrogram
            colors_rgb = calculate_spectrogram(eeg_data, 40)
            send_spectrogram_to_arduino(colors_rgb)

            # exit if we cannot read data from the device
            if nia_data.AccessDeniedError:
                sys.exit(1)

if __name__ == "__main__":
    app = web.application(urls, globals())

    # open the NIA, or exit with a failure code
    nia = NIA.NIA()
    if not nia.open():
        sys.exit(1)

    # start collecting data
    milliseconds = 50
    nia_data = NIA.NiaData(nia, milliseconds)

    # Create CSVWriter instance
    csv_writer = CSVWriter('nia_data_TESTTSTTS.csv')

    # kick-off processing data from the NIA
    updater = Updater(csv_writer)
    update_thread = threading.Thread(target=updater.update)
    update_thread.start()

    # run the app
    app.run()

    # when web.py exits, close out the NIA and exit gracefully
    running = False  # Arrêter l'exécution des threads
    update_thread.join()  # Attendre que les threads se terminent
    nia.close()
    sys.exit(0)
