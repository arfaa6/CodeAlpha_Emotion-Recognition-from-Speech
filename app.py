import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
import pickle
import sounddevice as sd
import soundfile as sf

# Import your exact preprocessor class from src folder
from src.preprocess import AudioPreprocessor

# Get absolute path directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and encoder safely
model_path = os.path.join(BASE_DIR, "model", "emotion_cnn_model.h5")
encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

print("Loading AI model...")
model = load_model(model_path)
with open(encoder_path, "rb") as f:
    le = pickle.load(f)
print("Model loaded successfully!")

# Initialize the exact preprocessor used during training
preprocessor = AudioPreprocessor()

def predict_emotion(audio_path):
    print(f"Starting prediction for: {audio_path}")
    
    # Extract features using the exact same pipeline as train.py
    features = preprocessor.extract_features(audio_path)
    if features is None:
        print("Feature extraction returned None!")
        return "Error", 0.0
    
    # Reshape to match model input requirement: (1, Height, Width, Channels)
    mfccs = np.expand_dims(features, axis=0)  # (1, 40, frames)
    mfccs = np.expand_dims(mfccs, axis=-1)   # (1, 40, frames, 1)
    
    print(f"Input shape to model: {mfccs.shape}")
    predictions = model.predict(mfccs, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_emotion = le.inverse_transform([predicted_index])[0]
    confidence = float(np.max(predictions))
    print(f"Prediction result: {predicted_emotion} ({confidence})")
    return predicted_emotion, confidence

# --- High-End Modern Dashboard UI ---
class SERApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AURA // Speech Emotion Intelligence")
        self.root.geometry("540x680")
        self.root.config(bg="#090D16")  # Deep space obsidian background

        # Top Header Banner
        header_frame = tk.Frame(root, bg="#090D16")
        header_frame.pack(pady=(30, 20))

        self.badge_lbl = tk.Label(
            header_frame, text="● SYSTEM ONLINE", 
            font=("Helvetica", 9, "bold"), bg="#090D16", fg="#34D399"
        )
        self.badge_lbl.pack(pady=(0, 5))

        title_lbl = tk.Label(
            header_frame, text="AURA INTELLIGENCE", 
            font=("Helvetica", 22, "bold"), bg="#090D16", fg="#F8FAFC"
        )
        title_lbl.pack()

        subtitle_lbl = tk.Label(
            header_frame, text="Advanced Neural Speech Emotion Recognition", 
            font=("Helvetica", 10), bg="#090D16", fg="#64748B"
        )
        subtitle_lbl.pack(pady=(2, 0))

        # Core Dashboard Card Container
        card = tk.Frame(root, bg="#131B2E", highlightbackground="#1E293B", highlightthickness=1)
        card.pack(padx=30, pady=10, fill="x")

        # Audio Status Section inside Card
        status_header = tk.Label(card, text="ACTIVE AUDIO SOURCE", font=("Helvetica", 9, "bold"), bg="#131B2E", fg="#94A3B8")
        status_header.pack(anchor="w", padx=20, pady=(15, 5))

        self.file_display = tk.Label(
            card, text="sample_audio.wav", 
            font=("Helvetica", 11), bg="#0B0F19", fg="#38BDF8", 
            padx=15, pady=12, anchor="w"
        )
        self.file_display.pack(padx=20, pady=(0, 15), fill="x")

        # Action Button Grid Frame
        btn_frame = tk.Frame(card, bg="#131B2E")
        btn_frame.pack(padx=20, pady=(0, 20), fill="x")

        # Sleek Interactive Buttons
        btn_style = {
            "font": ("Helvetica", 10, "bold"),
            "fg": "#FFFFFF",
            "height": 2,
            "bd": 0,
            "cursor": "hand2"
        }

        self.browse_btn = tk.Button(btn_frame, text="📁 Browse .wav", bg="#2563EB", activebackground="#1D4ED8", command=self.browse_file, **btn_style)
        self.browse_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        self.record_btn = tk.Button(btn_frame, text="🎙️ Record (3s)", bg="#059669", activebackground="#047857", command=self.record_audio, **btn_style)
        self.record_btn.pack(side="right", expand=True, fill="x", padx=(6, 0))

        # Prediction Trigger Button (Full Width Accent)
        self.predict_btn = tk.Button(
            card, text="✨ ANALYZE EMOTIONAL STATE", 
            font=("Helvetica", 11, "bold"), bg="#E11D48", fg="#FFFFFF", 
            activebackground="#BE123C", height=2, bd=0, cursor="hand2",
            command=self.run_prediction
        )
        self.predict_btn.pack(padx=20, pady=(0, 20), fill="x")

        # Dynamic Results Dashboard Panel
        result_card = tk.Frame(root, bg="#131B2E", highlightbackground="#334155", highlightthickness=1)
        result_card.pack(padx=30, pady=15, fill="x")

        result_title = tk.Label(result_card, text="ANALYSIS RESULT", font=("Helvetica", 9, "bold"), bg="#131B2E", fg="#94A3B8")
        result_title.pack(anchor="w", padx=20, pady=(15, 0))

        self.result_main = tk.Label(
            result_card, text="Awaiting input telemetry...", 
            font=("Helvetica", 14, "bold"), bg="#131B2E", fg="#FBBF24", pady=10
        )
        self.result_main.pack(padx=20)

        self.conf_lbl = tk.Label(
            result_card, text="Confidence score: —", 
            font=("Helvetica", 10), bg="#131B2E", fg="#64748B"
        )
        self.conf_lbl.pack(padx=20, pady=(0, 15))

        self.selected_path = "sample_audio.wav"

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if filename:
            self.selected_path = filename
            base_name = os.path.basename(filename)
            self.file_display.config(text=base_name, fg="#38BDF8")
            self.result_main.config(text="Sample loaded. Ready for analysis.", fg="#38BDF8")
            self.conf_lbl.config(text="Confidence score: Ready")

    def record_audio(self):
        self.file_display.config(text="🔴 Recording live audio... Speak now (3s)", fg="#F43F5E")
        self.result_main.config(text="Capturing microphone input...", fg="#F43F5E")
        self.conf_lbl.config(text="Recording in progress...")
        self.root.update()
        
        fs = 22050  
        seconds = 3
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()  
        
        self.selected_path = "live_recording.wav"
        sf.write(self.selected_path, myrecording, fs)
        
        self.file_display.config(text="live_recording.wav (Captured)", fg="#34D399")
        self.result_main.config(text="Recording saved successfully!", fg="#34D399")
        self.conf_lbl.config(text="Confidence score: Ready to evaluate")

    def run_prediction(self):
        if not os.path.exists(self.selected_path):
            messagebox.showerror("Error", "Please select or record a valid audio file first!")
            return
        
        self.result_main.config(text="Analyzing audio features...", fg="#38BDF8")
        self.conf_lbl.config(text="Running neural network...")
        self.root.update()
        
        emotion, conf = predict_emotion(self.selected_path)
        
        emotion_colors = {
            "happy": "#34D399", "angry": "#F43F5E", "sad": "#60A5FA", 
            "fearful": "#A78BFA", "disgusted": "#FB923C", "surprised": "#F472B6", 
            "calm": "#38BDF8", "neutral": "#94A3B8"
        }
        theme_color = emotion_colors.get(emotion.lower(), "#34D399")

        self.result_main.config(text=f"EMOTION: {emotion.upper()}", fg=theme_color)
        self.conf_lbl.config(text=f"Confidence Score: {conf * 100:.2f}% Match", fg="#F8FAFC")

if __name__ == "__main__":
    root = tk.Tk()
    app = SERApp(root)
    root.mainloop()