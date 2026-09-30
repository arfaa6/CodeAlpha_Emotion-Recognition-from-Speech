import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from preprocess import load_ravdess_dataset
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes):
    """Constructs a professional Deep CNN Architecture for SER."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    return model  # <--- Make sure this is properly indented INSIDE the function!

if __name__ == "__main__":
    DATASET_DIR = "data/" # Set path to your extracted RAVDESS folder
    
    X, y = load_ravdess_dataset(DATASET_DIR)
    
    # Reshape features to match CNN input requirement: (Samples, Height, Width, Channels)
    X = np.expand_dims(X, axis=-1)
    
    # Encode string labels to integers, then to categorical hot-vectors
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Save Label Encoder for inference
    os.makedirs("model", exist_ok=True)
    with open("model/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
        
    # Split Dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded)
    
    print(f"[INFO] Training set shape: {X_train.shape}")
    print(f"[INFO] Testing set shape: {X_test.shape}")
    
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=y_categorical.shape[1])
    model.summary()
 # Callbacks for optimal training
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True, verbose=1),
    ModelCheckpoint("model/emotion_cnn_model.h5", monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# Final Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n[RESULT] Test Accuracy: {accuracy * 100:.2f}%")