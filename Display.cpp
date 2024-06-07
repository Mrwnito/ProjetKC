#include <TFT_eSPI.h> // Bibliothèque pour l'écran TFT
#include <Arduino.h>
#include <vector>

TFT_eSPI tft = TFT_eSPI(); // Crée une instance de la classe TFT_eSPI
#define TFT_WIDTH  128
#define TFT_HEIGHT 160

std::vector<int> eeg_data;

void setup() {
  Serial.begin(921600); // Débit en bauds pour correspondre à votre configuration
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    updateEEGData(data);
    drawSpectrogram();
  }
}

void updateEEGData(String data) {
  eeg_data.clear();
  char* token = strtok((char*)data.c_str(), ",");
  while (token != NULL) {
    eeg_data.push_back(atoi(token));
    token = strtok(NULL, ",");
  }
}

void drawSpectrogram() {
  tft.fillScreen(TFT_BLACK);
  int num_data = eeg_data.size();
  for (int i = 0; i < num_data; i++) {
    int value = map(eeg_data[i], 0, 4096, 0, TFT_HEIGHT); // Adapter la plage de valeurs aux dimensions de l'écran
    tft.drawPixel(i % TFT_WIDTH, TFT_HEIGHT - value, TFT_WHITE);
  }
}
