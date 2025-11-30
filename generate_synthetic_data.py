#!/usr/bin/env python3
"""
Generate synthetic musical symbol images for training the OMR system
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random

def create_note_head(size=100, note_type='quarter'):
    """Create a note head image"""
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size//2, size//2)
    
    if note_type == 'quarter':
        # Filled oval
        cv2.ellipse(img, center, (size//8, size//6), 0, 0, 360, 255, -1)
    elif note_type == 'half':
        # Hollow oval
        cv2.ellipse(img, center, (size//8, size//6), 0, 0, 360, 255, 2)
    elif note_type == 'whole':
        # Hollow oval (slightly larger)
        cv2.ellipse(img, center, (size//7, size//5), 0, 0, 360, 255, 2)
    
    return img

def create_accidental(size=100, acc_type='sharp'):
    """Create an accidental symbol"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    if acc_type == 'sharp':
        # Draw sharp symbol
        cv2.line(img, (size//3, size//6), (size//3, size*5//6), 0, 2)
        cv2.line(img, (size*2//3, size//6), (size*2//3, size*5//6), 0, 2)
        cv2.line(img, (size//6, size//3), (size//2, size//3), 0, 2)
        cv2.line(img, (size//2, size*2//3), (size*5//6, size*2//3), 0, 2)
    elif acc_type == 'flat':
        # Draw flat symbol
        cv2.line(img, (size//4, size//6), (size//4, size*5//6), 0, 3)
        cv2.line(img, (size//4, size*5//6), (size*2//3, size*4//5), 0, 2)
    elif acc_type == 'natural':
        # Draw natural symbol
        cv2.line(img, (size//2, size//8), (size//2, size*7//8), 0, 2)
        cv2.line(img, (size//3, size//4), (size*2//3, size//4), 0, 2)
        cv2.line(img, (size//3, size*3//4), (size*2//3, size*3//4), 0, 2)
    
    return img

def create_note_stem(size=100):
    """Create a note stem"""
    img = np.zeros((size, size), dtype=np.uint8)
    # Vertical line
    cv2.line(img, (size//2, size//4), (size//2, size*3//4), 255, 2)
    return img

def create_chord(size=100):
    """Create a chord symbol"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    # Draw multiple note heads connected
    cv2.ellipse(img, (size//3, size//2), (size//10, size//8), 0, 0, 360, 0, -1)
    cv2.ellipse(img, (size*2//3, size//2), (size//10, size//8), 0, 0, 360, 0, -1)
    cv2.line(img, (size//3, size//2), (size//3, size//4), 0, 2)
    cv2.line(img, (size*2//3, size//2), (size*2//3, size//4), 0, 2)
    return img

def create_clef(size=100, clef_type='treble'):
    """Create a clef symbol"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    if clef_type == 'treble':
        # Simplified treble clef
        cv2.ellipse(img, (size//3, size//3), (size//8, size//6), 45, 0, 360, 0, -1)
        cv2.line(img, (size//3, size//3), (size//3, size*3//4), 0, 2)
        cv2.line(img, (size//4, size*3//4), (size//2, size*3//4), 0, 2)
    elif clef_type == 'bass':
        # Simplified bass clef
        cv2.circle(img, (size//2, size//2), size//6, 0, 2)
        cv2.line(img, (size//3, size//4), (size//3, size*3//4), 0, 2)
        cv2.line(img, (size*2//3, size//4), (size*2//3, size*3//4), 0, 2)
    
    return img

def create_bar_line(size=100):
    """Create a bar line"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    cv2.line(img, (size//2, size//8), (size//2, size*7//8), 0, 3)
    return img

def create_dot(size=100):
    """Create a dot symbol"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    cv2.circle(img, (size//2, size//2), size//12, 0, -1)
    return img

def create_rest(size=100, rest_type='quarter'):
    """Create a rest symbol"""
    img = np.ones((size, size), dtype=np.uint8) * 255
    
    if rest_type == 'quarter':
        # Quarter rest
        cv2.line(img, (size//3, size//4), (size*2//3, size//4), 0, 3)
        cv2.line(img, (size//3, size//4), (size//3, size//2), 0, 2)
        cv2.ellipse(img, (size//2, size*2//3), (size//6, size//8), 0, 0, 180, 0, -1)
    elif rest_type == 'half':
        # Half rest
        cv2.rectangle(img, (size//4, size//3), (size*3//4, size//2), 0, -1)
    elif rest_type == 'whole':
        # Whole rest
        cv2.rectangle(img, (size//4, size//2), (size*3//4, size*2//3), 0, -1)
    
    return img

def add_noise_and_variation(img, variation_level=0.3):
    """Add noise and variation to make images more realistic"""
    # Add Gaussian noise
    noise = np.random.normal(0, variation_level * 50, img.shape)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # Add slight rotation
    angle = random.uniform(-5, 5)
    center = (img.shape[1]//2, img.shape[0]//2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(noisy_img, matrix, (img.shape[1], img.shape[0]))
    
    # Add slight scale variation
    scale = random.uniform(0.9, 1.1)
    scaled = cv2.resize(rotated, None, fx=scale, fy=scale)
    
    # Ensure consistent size
    if scaled.shape != img.shape:
        scaled = cv2.resize(scaled, (img.shape[1], img.shape[0]))
    
    return scaled

def generate_synthetic_data():
    """Generate synthetic training data for all classes"""
    output_dir = 'train_data/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define what symbols to generate
    symbols_to_generate = {
        # Notes
        'c1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'd1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'e1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'f1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'g1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'a1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'b1': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'c2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'd2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'e2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'f2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'g2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'a2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        'b2': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        
        # Accidentals
        'sharp': lambda: add_noise_and_variation(create_accidental(100, 'sharp')),
        'flat': lambda: add_noise_and_variation(create_accidental(100, 'flat')),
        'natural': lambda: add_noise_and_variation(create_accidental(100, 'natural')),
        
        # Other symbols
        'clef': lambda: add_noise_and_variation(create_clef(100, 'treble')),
        'bar': lambda: add_noise_and_variation(create_bar_line(100)),
        'dot': lambda: add_noise_and_variation(create_dot(100)),
        'chord': lambda: add_noise_and_variation(create_chord(100)),
        
        # Note durations
        '1': lambda: add_noise_and_variation(create_rest(100, 'whole')),
        '2': lambda: add_noise_and_variation(create_rest(100, 'half')),
        '4': lambda: add_noise_and_variation(create_note_head(100, 'quarter')),
        '8': lambda: add_noise_and_variation(create_rest(100, 'quarter')),
        '16': lambda: add_noise_and_variation(create_rest(100, 'quarter')),
        '32': lambda: add_noise_and_variation(create_rest(100, 'quarter')),
    }
    
    # Generate 20 images per symbol
    images_per_symbol = 20
    
    for symbol_name, generator_func in symbols_to_generate.items():
        symbol_dir = os.path.join(output_dir, symbol_name)
        os.makedirs(symbol_dir, exist_ok=True)
        
        print(f"Generating {images_per_symbol} images for {symbol_name}")
        
        for i in range(images_per_symbol):
            img = generator_func()
            img_path = os.path.join(symbol_dir, f"{symbol_name}_{i}.png")
            cv2.imwrite(img_path, img)
    
    print("Synthetic data generation completed!")

if __name__ == "__main__":
    generate_synthetic_data()