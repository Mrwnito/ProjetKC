import serial
import time
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Initialiser la connexion série
serial_port = serial.Serial('COM5', 921600)

# Générer des données EEG simulées pour le spectrogramme
fs = 40
t = np.linspace(0, 10, 10 * fs)
eeg_data = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 8 * t)

# Calculer le spectrogramme
f, t, Sxx = spectrogram(eeg_data, fs, nperseg=128)
Sxx_log = 10 * np.log10(Sxx)  # Convertir en échelle logarithmique

norm = Normalize(vmin=np.min(Sxx_log), vmax=np.max(Sxx_log))
cmap = plt.get_cmap('viridis')
colors_rgb = cmap(norm(Sxx_log))
colors_rgb = (colors_rgb[:, :, :3] * 255).astype(np.uint8)  # Convertir en valeurs RGB

# Envoyer les données de spectrogramme au port série
height, width, _ = colors_rgb.shape
while True:
    for j in range(height):
        for i in range(width):
            r, g, b = colors_rgb[j, i]
            color_string = bytes([r, g, b])
            serial_port.write(color_string)
            time.sleep(0.001)  # Ajuster la vitesse d'envoi si nécessaire
    print("Données du spectrogramme envoyées")
    time.sleep(1)
