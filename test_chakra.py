import time
import threading
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nia as NIA
import sys

# Configuration des filtres
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Calcul des amplitudes des bandes de fréquences
def calculate_amplitudes(eeg_data, fs):
    if eeg_data.size == 0:
        return None, None, None, None, None

    delta = bandpass_filter(eeg_data, 0.5, 3.9, fs)
    theta = bandpass_filter(eeg_data, 4.0, 7.9, fs)
    alpha = bandpass_filter(eeg_data, 8.0, 11.9, fs)
    beta = bandpass_filter(eeg_data, 12.0, 29.9, fs)
    gamma = bandpass_filter(eeg_data, 30.0, 99.9, fs)

    delta_amp = np.mean(np.abs(delta))
    theta_amp = np.mean(np.abs(theta))
    alpha_amp = np.mean(np.abs(alpha))
    beta_amp = np.mean(np.abs(beta))
    gamma_amp = np.mean(np.abs(gamma))

    return delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp

# Calculate relative changes
def calculate_relative_changes(current, previous):
    changes = {}
    for key in current:
        changes[key] = abs(current[key] - previous[key]) / previous[key] if previous[key] != 0 else 0
    return changes

# Mapping des fréquences aux chakras
def map_frequencies_to_chakras(delta, theta, alpha, beta, gamma):
    chakra_colors = {
        'Muladhara': np.array([1, 0, 0]),       # red
        'Svadhisthana': np.array([1, 0.5, 0]),  # orange
        'Manipura': np.array([1, 1, 0]),        # yellow
        'Anahata': np.array([0, 1, 0]),         # green
        'Vishuddha': np.array([0, 0, 1]),       # blue
        'Ajna': np.array([0.29, 0, 0.51]),      # indigo
        'Sahasrara': np.array([0.93, 0.51, 0.93]) # violet
    }

    chakra_activation = {
        'Muladhara': delta,
        'Svadhisthana': theta,
        'Manipura': beta,
        'Anahata': alpha,
        'Vishuddha': alpha,
        'Ajna': theta,
        'Sahasrara': gamma
    }
    return chakra_activation, chakra_colors

# Combine multiple colors
def combine_colors(changes, chakra_colors):
    combined_color = np.array([0.0, 0.0, 0.0])
    for chakra, change in changes.items():
        if change > 0:
            combined_color += np.array(chakra_colors[chakra]) * change
    combined_color = np.clip(combined_color, 0, 1)
    return combined_color

# Smoothly transition to the new color
def smooth_transition_color(current_color, target_color, transition_speed=0.1):
    return current_color + (target_color - current_color) * transition_speed

# Visualisation des chakras
def update_circle(ax, radius, color):
    ax.clear()
    circle = plt.Circle((0.5, 0.5), radius, color=color, alpha=0.5)
    ax.add_artist(circle)
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.4, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')

def animate(i, updater, ax):
    if updater.data_to_plot:
        chakra_activation, chakra_colors, changes = updater.data_to_plot
        target_color = combine_colors(changes, chakra_colors)
        updater.current_color = smooth_transition_color(updater.current_color, target_color)
        overall_activity = sum(changes.values()) / len(changes)  # Average change
        radius = overall_activity * 0.05  # Smaller radius changes
        update_circle(ax, radius, updater.current_color)

def visualize_dynamic(updater):
    fig, ax = plt.subplots()
    ani = FuncAnimation(fig, animate, fargs=(updater, ax), interval=100)  # Update every 100ms
    plt.show()

# Classe pour gérer la collecte des données EEG
class EEGData:
    def __init__(self, sample_interval_ms):
        self.nia = NIA.NIA()
        if not self.nia.open():
            sys.exit("Failed to open NIA device")
        self.nia_data = NIA.NiaData(self.nia, sample_interval_ms)  # Collecte des données à l'intervalle spécifié

    def get_data(self):
        self.nia_data.get_data()
        return np.array(self.nia_data.Raw_Data)

class Updater:
    def __init__(self, sample_interval_ms):
        self.eeg_data_source = EEGData(sample_interval_ms)
        self.fs = 256  # Set fixed sampling rate
        self.collect_interval = sample_interval_ms / 1000.0  # Intervalle de collecte configuré (en secondes)
        self.data_to_plot = None
        self.previous_activation = {
            'Muladhara': 0,
            'Svadhisthana': 0,
            'Manipura': 0,
            'Anahata': 0,
            'Vishuddha': 0,
            'Ajna': 0,
            'Sahasrara': 0
        }
        self.current_color = np.array([0.0, 0.0, 0.0])

    def update(self):
        global running
        while running:
            eeg_data = self.eeg_data_source.get_data()
            delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp = calculate_amplitudes(eeg_data, self.fs)
            chakra_activation, chakra_colors = map_frequencies_to_chakras(delta_amp, theta_amp, alpha_amp, beta_amp, gamma_amp)
            changes = calculate_relative_changes(chakra_activation, self.previous_activation)
            self.previous_activation = chakra_activation
            self.data_to_plot = (chakra_activation, chakra_colors, changes)
            print(f"Delta: {delta_amp}, Theta: {theta_amp}, Alpha: {alpha_amp}, Beta: {beta_amp}, Gamma: {gamma_amp}")
            print(f"Relative Changes: {changes}")
            time.sleep(self.collect_interval)

if __name__ == "__main__":
    running = True
    sample_interval_ms = 10
    updater = Updater(sample_interval_ms)
    update_thread = threading.Thread(target=updater.update)
    update_thread.start()

    try:
        visualize_dynamic(updater)
    except KeyboardInterrupt:
        running = False
        update_thread.join()
        updater.eeg_data_source.nia.close()
        sys.exit(0)
