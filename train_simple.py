#!/usr/bin/env python3
"""
Simplified training script for Mozart OMR system
"""

import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import os
from glob import glob

# Configuration
dataset_path = 'train_data/data'
target_img_size = (100, 100)

def extract_hog_features(img):
    """Extract HOG features from image"""
    img = cv2.resize(img, target_img_size)
    win_size = (100, 100)
    cell_size = (5, 5)
    block_size = (15, 15)  # 3x3 cells
    block_stride = (5, 5)
    nbins = 12
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    return h.flatten()

def load_dataset():
    """Load and preprocess dataset"""
    features = []
    labels = []
    
    print("Loading dataset...")
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_name in subdirs:
        class_path = os.path.join(dataset_path, class_name)
        image_files = glob(os.path.join(class_path, "*.png")) + \
                     glob(os.path.join(class_path, "*.jpg")) + \
                     glob(os.path.join(class_path, "*.jpeg"))
        
        print(f"Processing {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is not None:
                features.append(extract_hog_features(img))
                labels.append(class_name)
    
    print(f"Total images loaded: {len(features)}")
    return np.array(features, dtype=np.float32), np.array(labels)

def train_model():
    """Train the neural network model"""
    # Load dataset
    features, labels = load_dataset()
    
    if len(features) < 10:
        print("ERROR: Not enough training data!")
        print("Please add more images to train_data/data/ subdirectories")
        return None
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=42)
    
    # Create and train model
    print("Training neural network...")
    model = MLPClassifier(
        hidden_layer_sizes=(200, 100),
        max_iter=1000,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save model and label encoder
    os.makedirs('trained_models', exist_ok=True)
    with open('trained_models/nn_simple_model.sav', 'wb') as f:
        pickle.dump(model, f)
    
    with open('trained_models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    print("Model saved successfully!")
    return model, le

if __name__ == "__main__":
    result = train_model()
    if result:
        model, le = result
        print("Training completed successfully!")
    else:
        print("Training failed - insufficient data")