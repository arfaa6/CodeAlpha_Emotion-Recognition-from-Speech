import os
import numpy as np
import librosa

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=3.0, n_mfcc=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.sample_length = int(sample_rate * duration)
        self.n_mfcc = n_mfcc

    def pad_or_truncate(self, y):
        """Ensures uniform audio length for batch processing."""
        if len(y) == 0:
            y = np.zeros(self.sample_length)
        if len(y) < self.sample_length:
            y = np.pad(y, (0, self.sample_length - len(y)), mode='constant')
        else:
            y = y[:self.sample_length]
        return y

    def extract_features(self, file_path):
        """Extracts MFCC and returns a 2D matrix representing the audio feature map."""
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            if len(y) == 0:
                return None
            y = self.pad_or_truncate(y)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            if mfccs.size == 0 or np.max(mfccs) == np.min(mfccs):
                return None
            mfccs = librosa.util.normalize(mfccs)
            return mfccs
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def load_ravdess_dataset(dataset_path):
    """
    Parses RAVDESS naming convention:
    e.g., 03-01-05-01-02-01-12.wav -> 3rd section is Emotion (05 = angry)
    """
    preprocessor = AudioPreprocessor()
    features, labels = [], []
    
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    print("[INFO] Scanning dataset directory...")
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        raise ValueError(f"Directory '{dataset_path}' is empty or does not exist! Please place your RAVDESS actor folders inside 'data/'.")

    file_count = 0
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_count += 1
                parts = file.split('-')
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    file_path = os.path.join(root, file)
                    mfccs = preprocessor.extract_features(file_path)
                    if mfccs is not None:
                        features.append(mfccs)
                        labels.append(emotion_map[emotion_code])

    print(f"[INFO] Found {file_count} total audio files. Successfully extracted features for {len(features)} files.")
    
    if len(features) == 0:
        raise ValueError("No valid audio features were extracted. Check if your RAVDESS dataset is properly extracted inside the 'data/' folder.")

    return np.array(features), np.array(labels)