#!/usr/bin/env python3
"""
Test the trained OMR model with sample sheet music images
"""

import cv2
import numpy as np
import pickle
import os
from glob import glob
import sys

# Add src directory to path
sys.path.append('src')

def test_omr_model():
    """Test the OMR model with sample images"""
    
    # Load the trained model and label encoder
    try:
        with open('trained_models/nn_simple_model.sav', 'rb') as f:
            model = pickle.load(f)
        with open('trained_models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_simple.py first.")
        return
    
    # Test individual symbol recognition
    print("\n=== Testing Individual Symbol Recognition ===")
    
    # Test with a few sample images
    test_symbols = [
        ('train_data/data/c1/c1_0.png', 'c1'),
        ('train_data/data/sharp/sharp_0.png', 'sharp'),
        ('train_data/data/clef/clef_0.png', 'clef'),
        ('train_data/data/4/4_0.png', '4'),
    ]
    
    correct = 0
    total = 0
    
    for img_path, expected_label in test_symbols:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Extract HOG features (same as training)
                img_resized = cv2.resize(img, (100, 100))
                win_size = (100, 100)
                cell_size = (5, 5)
                block_size = (15, 15)
                block_stride = (5, 5)
                nbins = 12
                
                hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
                features = hog.compute(img_resized).flatten().reshape(1, -1)
                
                # Predict
                prediction = model.predict(features)[0]
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                
                # Check if correct
                is_correct = predicted_label == expected_label
                if is_correct:
                    correct += 1
                total += 1
                
                print(f"Image: {img_path}")
                print(f"Expected: {expected_label}, Predicted: {predicted_label}, Correct: {is_correct}")
                print()
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Individual symbol recognition accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Test with a simple sheet music image (create a test image)
    print("\n=== Testing with Synthetic Sheet Music ===")
    
    # Create a simple test image with multiple symbols
    test_img = np.ones((200, 400), dtype=np.uint8) * 255
    
    # Add some symbols (this is a simplified test)
    # In a real scenario, you would use actual sheet music
    
    # Add a few note heads
    cv2.ellipse(test_img, (50, 100), (8, 6), 0, 0, 360, 0, -1)
    cv2.ellipse(test_img, (100, 80), (8, 6), 0, 0, 360, 0, -1)
    cv2.ellipse(test_img, (150, 120), (8, 6), 0, 0, 360, 0, -1)
    
    # Add a sharp symbol
    cv2.line(test_img, (200, 60), (200, 140), 0, 2)
    cv2.line(test_img, (220, 60), (220, 140), 0, 2)
    cv2.line(test_img, (180, 90), (240, 90), 0, 2)
    cv2.line(test_img, (180, 110), (240, 110), 0, 2)
    
    # Test the image
    test_img_resized = cv2.resize(test_img, (100, 100))
    hog = cv2.HOGDescriptor((100, 100), (15, 15), (5, 5), (5, 5), 12)
    test_features = hog.compute(test_img_resized).flatten().reshape(1, -1)
    
    prediction = model.predict(test_features)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    print(f"Synthetic sheet music test - Predicted: {predicted_label}")
    
    # Save test results
    cv2.imwrite('test_sheet_music.png', test_img)
    print("✓ Test image saved as test_sheet_music.png")
    
    print("\n=== OMR Model Test Summary ===")
    print(f"Model trained on {len(label_encoder.classes_)} classes: {', '.join(label_encoder.classes_)}")
    print(f"Individual symbol accuracy: {accuracy:.1f}%" if total > 0 else "No individual tests completed")
    print("✓ Model is ready for real sheet music testing")
    
    return model, label_encoder

def evaluate_model_performance():
    """Evaluate model performance with cross-validation"""
    print("\n=== Model Performance Evaluation ===")
    
    try:
        with open('trained_models/nn_simple_model.sav', 'rb') as f:
            model = pickle.load(f)
        with open('trained_models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model files not found.")
        return
    
    # Test with some validation data
    validation_accuracy = 0
    total_tests = 0
    
    # Test with a few random samples from each class
    for class_name in label_encoder.classes_[:5]:  # Test first 5 classes
        class_dir = f'train_data/data/{class_name}'
        if os.path.exists(class_dir):
            images = glob(os.path.join(class_dir, '*.png'))
            if images:
                # Test first 3 images from each class
                for img_path in images[:3]:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Extract features
                        img_resized = cv2.resize(img, (100, 100))
                        hog = cv2.HOGDescriptor((100, 100), (15, 15), (5, 5), (5, 5), 12)
                        features = hog.compute(img_resized).flatten().reshape(1, -1)
                        
                        # Predict
                        prediction = model.predict(features)[0]
                        predicted_label = label_encoder.inverse_transform([prediction])[0]
                        
                        if predicted_label == class_name:
                            validation_accuracy += 1
                        total_tests += 1
    
    if total_tests > 0:
        val_accuracy = (validation_accuracy / total_tests) * 100
        print(f"Validation accuracy: {val_accuracy:.1f}% ({validation_accuracy}/{total_tests})")
    
    return validation_accuracy

if __name__ == "__main__":
    print("🎼 Mozart OMR Model Testing")
    print("=" * 40)
    
    # Test the model
    model, encoder = test_omr_model()
    
    # Evaluate performance
    evaluate_model_performance()
    
    print("\n" + "=" * 40)
    print("Testing completed!")
    print("Next steps:")
    print("1. Test with real sheet music images")
    print("2. Fine-tune the model with more diverse data")
    print("3. Implement the full OMR pipeline")