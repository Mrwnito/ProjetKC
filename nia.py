import usb.core
import usb.backend.libusb1
import numpy as np
import sys
import math
import threading

# Charger le backend libusb1
backend = usb.backend.libusb1.get_backend()
running = True  # Indicateur pour contrôler l'exécution des threads
class DeviceDescriptor:
    def __init__(self, vendor_id, product_id, interface_id):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.interface_id = interface_id

    def get_device(self):
        return usb.core.find(idVendor=self.vendor_id, idProduct=self.product_id, backend=backend)

class NIA:
    VENDOR_ID = 0x1234
    PRODUCT_ID = 0x0000
    INTERFACE_ID = 0
    BULK_IN_EP = 0x81  # Endpoint modifié selon votre périphérique
    BULK_OUT_EP = 0x01  # Endpoint modifié selon votre périphérique
    PACKET_LENGTH = 0x40

    device_descriptor = DeviceDescriptor(VENDOR_ID, PRODUCT_ID, INTERFACE_ID)

    def __init__(self):
        self.device = self.device_descriptor.get_device()
        self.handle = None

    def open(self):
        self.device = self.device_descriptor.get_device()
        if not self.device:
            print("Failed to open NIA device. Cable isn't plugged in", file=sys.stderr)
            return False
        try:
            self.device.set_configuration()
            self.handle = self.device
            try:
                usb.util.claim_interface(self.handle, self.device_descriptor.interface_id)
            except Exception as e:
                print(e, file=sys.stderr)
                return False
        except usb.core.USBError as err:
            print(err, file=sys.stderr)
            return False
        return True

    def close(self):
        try:
            usb.util.release_interface(self.handle, self.device_descriptor.interface_id)
            usb.util.dispose_resources(self.handle)
        except Exception as err:
            print(err, file=sys.stderr)
        self.handle, self.device = None, None

    def bulk_read(self):
        global running
        if not running:
            return np.zeros(64)  # Retourner des données vides si l'exécution est arrêtée
        read_bytes = self.handle.read(self.BULK_IN_EP, self.PACKET_LENGTH, timeout=25)
        return read_bytes

class NiaData:
    def __init__(self, nia, milliseconds):
        self.Points = milliseconds / 2
        self.Processed_Data = np.ones(4096, dtype=np.uint32)
        self.Raw_Data = np.zeros(10, dtype=np.uint32)
        self.Fourier_Data = np.zeros((140, 160), dtype=np.int8)
        self.AccessDeniedError = False
        self.nia = nia

    def get_data(self):
        global running
        Raw_Data = np.array([])
        try:
            for _ in range(int(self.Points)):
                if not running:
                    break  # Sortir de la boucle si l'exécution est arrêtée
                data = self.nia.bulk_read()
                p = int(data[54])
                temp = np.zeros(p, dtype=np.uint32)
                for col in range(p):
                    temp[col] = data[col * 3 + 2] * 65536 + data[col * 3 + 1] * 256 + data[col * 3]
                Raw_Data = np.append(Raw_Data, temp)
                #print(f"Data read from NIA: {data}")  # Ajoutez cette ligne pour vérifier les données brutes lues
        except usb.core.USBError as err:
            print("Failed to access NIA device: Access Denied", file=sys.stderr)
            print("If you're on GNU/Linux, see README Troubleshooting section for details", file=sys.stderr)
            self.AccessDeniedError = True
        self.Processed_Data = np.append(self.Processed_Data, Raw_Data)[-4096:-1]
        self.Raw_Data = Raw_Data
        #print(f"Raw_Data collected: {self.Raw_Data}")  # Ajoutez cette ligne pour vérifier les données collectées

    def waveform(self):
        filter_over = 30
        data = np.fft.fftn(self.Processed_Data[::8])
        data[filter_over:-filter_over] = 0
        data = np.fft.ifft(data).real  # Utiliser uniquement la partie réelle
        x_max = max(data) * 1.1
        x_min = min(data) * 0.9
        data = (140 * (data - x_min) / (x_max - x_min))
        wave = np.ones((140, 410), dtype=np.int8)
        wave = np.dstack((wave * 0, wave * 0, wave * 51))
        for i in range(410):
            wave_data_index = data[i + 102]
            if not np.isnan(wave_data_index):
                wave[int(wave_data_index), i, :] = [0, 204, 255]
        return wave.tostring()

    def fourier(self, data):
        self.Fourier_Data[1:140, :] = self.Fourier_Data[0:139, :]
        x = abs(np.fft.fftn(data.Processed_Data * np.hanning(len(data.Processed_Data))))[4:44]
        x_max = max(x)
        x_min = min(x)
        x = (255 * (x - x_min) / (x_max - x_min))
        pointer = np.zeros((160), dtype=np.int8)
        pointer[(np.argmax(x)) * 4:(np.argmax(x)) * 4 + 4] = 255
        y = np.vstack((x, x, x, x))
        y = np.ravel(y, 'F')
        self.Fourier_Data[5, :] = y
        self.Fourier_Data[0:4, :] = np.vstack((pointer, pointer, pointer, pointer))
        fingers = []
        waves = (6, 9, 12, 15, 20, 25, 30)
        for i in range(6):
            finger_sum = sum(x[waves[i]:waves[i + 1]]) / 100
            if not np.isnan(finger_sum):
                fingers.append(finger_sum)
            else:
                fingers.append(0)
        return self.Fourier_Data.tostring(), fingers
