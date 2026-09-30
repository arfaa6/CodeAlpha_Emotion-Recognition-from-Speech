# AURA // Speech Emotion Intelligence

A high-performance Speech Emotion Recognition (SER) desktop application powered by a custom Deep Convolutional Neural Network (CNN), built with TensorFlow/Keras and featuring a modern, dark-mode graphical user interface (Tkinter).

![System Online](https://img.shields.io/badge/System-Online-34D399?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)

---

## 🌟 Features
* **Modern GUI Dashboard:** A sleek, deep-space obsidian dashboard built with custom Tkinter styling.
* **Dual Input Modes:** 
  * 📁 **Browse & Analyze:** Upload any `.wav` audio file for instant inference.
  * 🎙️ **Live Recording:** Record a 3-second live microphone sample in real-time.
* **Deep Neural Architecture:** Utilizes a multi-layer 2D Convolutional Neural Network trained on MFCC (Mel-Frequency Cepstral Coefficients) audio feature maps.
* **Robust Preprocessing:** Standardized audio length padding, truncation, and normalization mirroring training conditions.

---

## 📂 Project Structure
```text
speech-emotion-recognition/
│
├── data/                       # RAVDESS dataset folder (ignored in git)
├── model/                      
│   ├── emotion_cnn_model.h5    # Trained CNN weights
│   └── label_encoder.pkl       # Target emotion label encoder
│
├── src/
│   └── preprocess.py           # AudioPreprocessor class & feature extraction
│
├── app.py                      # Main Tkinter GUI application
├── train.py                    # Model training script
├── sample_audio.wav            # Default test sample
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation