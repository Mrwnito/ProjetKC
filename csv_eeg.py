import time
import threading
import numpy as np
from scipy.signal import butter, lfilter
import csv
import nia as NIA
import sys

# Configuration des filtres
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
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

def calculate_amplitudes(eeg_mean, fs):
    delta = bandpass_filter(eeg_mean, 0.5, 3.9, fs)
    theta = bandpass_filter(eeg_mean, 4.0, 7.9, fs)
    alpha = bandpass_filter(eeg_mean, 8.0, 11.9, fs)
    beta = bandpass_filter(eeg_mean, 12.0, 29.9, fs)
    gamma = bandpass_filter(eeg_mean, 30.0, 99.9, fs)

    delta_amp = np.mean(np.abs(delta))
    theta_amp = np.mean(np.abs(theta))
    alpha_amp = np.mean(np.abs(alpha))
    beta_amp = np.mean(np.abs(beta))
    gamma_amp = np.mean(np.abs(gamma))

    return delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp

class CSVWriter:
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = ['timestamp', 'eeg_data', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        self.write_header()

    def write_header(self):
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_row(self, data):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(data)

class EEGData:
    def __init__(self, sample_interval_ms):
        self.nia = NIA.NIA()
        if not self.nia.open():
            sys.exit("Failed to open NIA device")
        self.nia_data = NIA.NiaData(self.nia, sample_interval_ms)

    def get_data(self):
        self.nia_data.get_data()
        return np.array(self.nia_data.Raw_Data)

class Updater:
    def __init__(self, sample_interval_ms, csv_writer):
        self.eeg_data_source = EEGData(sample_interval_ms)
        self.fs = 1000
        self.collect_interval = sample_interval_ms / 1000.0
        self.csv_writer = csv_writer

    def update(self):
        global running
        while running:
            eeg_data = self.eeg_data_source.get_data()
            eeg_mean = np.mean(eeg_data)
            print(f"eeg_mean : {eeg_mean}")
            delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp = calculate_amplitudes(eeg_data, self.fs)
            timestamp = time.time()
            row = {
                'timestamp': timestamp,
                'eeg_data': eeg_mean,
                'delta': delta_amp,
                'theta': theta_amp,
                'alpha': alpha_amp,
                'beta': beta_amp,
                'gamma': gamma_amp
            }
            self.csv_writer.write_row(row)
            time.sleep(self.collect_interval)

if __name__ == "__main__":
    running = True
    sample_interval_ms = 1  # For 512 Hz sampling rate
    csv_writer = CSVWriter('eeg_data_TEST.csv')
    updater = Updater(sample_interval_ms, csv_writer)
    update_thread = threading.Thread(target=updater.update)
    update_thread.start()

    try:
        while True:
            time.sleep(sample_interval_ms / 1000.0)
    except KeyboardInterrupt:
        running = False
        update_thread.join()
        updater.eeg_data_source.nia.close()
        sys.exit(0)
