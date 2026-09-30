import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
import pickle
import tensorflow as tf
from preprocess import AudioPreprocessor

def predict_emotion(audio_path):
    # Load model and label encoder
    model = tf.keras.models.load_model("model/emotion_cnn_model.h5")
    with open("model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
        
    preprocessor = AudioPreprocessor()
    mfccs = preprocessor.extract_features(audio_path)
    
    if mfccs is None:
        return "Could not process audio."
        
    # Expand dimensions to match model input specification
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    predictions = model.predict(mfccs)
    predicted_index = np.argmax(predictions)
    predicted_emotion = le.inverse_transform([predicted_index])[0]
    confidence = float(np.max(predictions))
    
    return predicted_emotion, confidence

if __name__ == "__main__":
    test_file = "sample_audio.wav"
    if os.path.exists(test_file):
        emotion, conf = predict_emotion(test_file)

        # This creates a pop-up window
        root = tk.Tk()
        root.withdraw() # Hides the small extra background window
        messagebox.showinfo(
            "Speech Emotion Recognition", 
            f"Detected Emotion: {emotion.upper()}\nConfidence: {conf * 100:.2f}%"
        )
    else:
        print("Please provide a valid path to an audio file for testing.")