import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
import pickle
import sounddevice as sd
import soundfile as sf

from src.preprocess import AudioPreprocessor

# Get absolute path directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "emotion_cnn_model.h5")
encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

print("Loading AI model...")
model = load_model(model_path)
with open(encoder_path, "rb") as f:
    le = pickle.load(f)
print("Model loaded successfully!")

preprocessor = AudioPreprocessor()

def predict_emotion(audio_path):
    features = preprocessor.extract_features(audio_path)
    if features is None:
        return "Error", 0.0
    mfccs = np.expand_dims(features, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    predictions = model.predict(mfccs, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_emotion = le.inverse_transform([predicted_index])[0]
    confidence = float(np.max(predictions))
    return predicted_emotion, confidence

# --- AURA Professional Pastel UI ---
class AURAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA // Speech Emotion Recognition")
        self.root.geometry("540x520")
        self.root.config(bg="#EDECDB")  # Lychee Background

        # Minimalist Clean Header
        header_frame = tk.Frame(root, bg="#EDECDB")
        header_frame.pack(anchor="w", padx=30, pady=(25, 10))

        tk.Label(
            header_frame, text="AURA INTELLIGENCE", 
            font=("Helvetica", 14, "bold"), bg="#EDECDB", fg="#2D3748"
        ).pack(anchor="w")

        tk.Label(
            header_frame, text="Neural Speech Emotion Analysis Pipeline", 
            font=("Helvetica", 9), bg="#EDECDB", fg="#718096"
        ).pack(anchor="w", pady=(2, 0))

        # Main Structural Card
        card = tk.Frame(root, bg="#FFFFFF", highlightbackground="#A7BCBD", highlightthickness=1)
        card.pack(padx=25, pady=5, fill="both", expand=True)

        # Input Source Section
        input_section = tk.Frame(card, bg="#FFFFFF")
        input_section.pack(padx=25, pady=(20, 10), fill="x")

        tk.Label(
            input_section, text="TARGET AUDIO SOURCE", 
            font=("Helvetica", 8, "bold"), bg="#FFFFFF", fg="#6BB1AD"
        ).pack(anchor="w", pady=(0, 6))

        self.file_display_lbl = tk.Label(
            input_section, text="sample_audio.wav", 
            font=("Courier", 10), bg="#EDECDB", fg="#2D3748", anchor="w", padx=10, pady=8
        )
        self.file_display_lbl.pack(fill="x")

        # Action Buttons Row
        action_row = tk.Frame(card, bg="#FFFFFF")
        action_row.pack(padx=25, pady=(10, 15), fill="x")

        self.browse_btn = tk.Button(
            action_row, text="Browse .wav", font=("Helvetica", 9, "bold"),
            bg="#6BB1AD", fg="#FFFFFF", activebackground="#579B97", bd=0, pady=8, cursor="hand2",
            command=self.browse_file
        )
        self.browse_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.record_btn = tk.Button(
            action_row, text="Record (3s)", font=("Helvetica", 9, "bold"),
            bg="#E5A9A9", fg="#2D3748", activebackground="#D49898", bd=0, pady=8, cursor="hand2",
            command=self.record_audio
        )
        self.record_btn.pack(side="right", fill="x", expand=True, padx=(6, 0))

        # Primary Run Button
        self.analyze_btn = tk.Button(
            card, text="RUN ANALYSIS", 
            font=("Helvetica", 10, "bold"), bg="#E6748E", fg="#FFFFFF", 
            activebackground="#D05F7B", bd=0, pady=12, cursor="hand2",
            command=self.run_prediction
        )
        self.analyze_btn.pack(padx=25, pady=(5, 20), fill="x")

        # Results Dashboard Box
        result_box = tk.Frame(card, bg="#EDECDB", padx=15, pady=12)
        result_box.pack(padx=25, pady=(0, 25), fill="x")

        tk.Label(
            result_box, text="OUTPUT RESULT", 
            font=("Helvetica", 7, "bold"), bg="#EDECDB", fg="#6BB1AD"
        ).pack(anchor="w")

        self.status_lbl = tk.Label(
            result_box, text="Awaiting Input...", 
            font=("Helvetica", 12, "bold"), bg="#EDECDB", fg="#2D3748"
        )
        self.status_lbl.pack(anchor="w", pady=(2, 0))

        self.conf_lbl = tk.Label(
            result_box, text="Confidence: —", 
            font=("Helvetica", 9), bg="#EDECDB", fg="#718096"
        )
        self.conf_lbl.pack(anchor="w", pady=(2, 0))

        self.selected_path = "sample_audio.wav"

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if filename:
            self.selected_path = filename
            base_name = os.path.basename(filename)
            self.file_display_lbl.config(text=base_name)

    def record_audio(self):
        self.file_display_lbl.config(text="Recording microphone input...")
        self.root.update()
        
        fs = 22050  
        seconds = 3
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  
        
        self.selected_path = "live_recording.wav"
        sf.write(self.selected_path, myrecording, fs)
        
        self.file_display_lbl.config(text="live_recording.wav")

    def run_prediction(self):
        if not os.path.exists(self.selected_path):
            messagebox.showerror("Error", "Please select or record a valid audio file first!")
            return
        
        self.status_lbl.config(text="Analyzing audio...", fg="#718096")
        self.conf_lbl.config(text="Confidence: Calculating...")
        self.root.update()
        
        emotion, conf = predict_emotion(self.selected_path)
        
        # Update results with professional styling
        self.status_lbl.config(text=f"Emotion: {emotion.upper()}", fg="#6BB1AD")
        self.conf_lbl.config(text=f"Confidence: {conf * 100:.2f}% Match")

if __name__ == "__main__":
    root = tk.Tk()
    app = AURAApp(root)
    root.mainloop()