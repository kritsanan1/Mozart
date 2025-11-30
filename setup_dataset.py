#!/usr/bin/env python3
"""
Dataset Setup Script for Mozart OMR
Processes dissertation pages to extract musical symbols for training
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

class DatasetProcessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_path = self.output_path / "train_data" / "data"
        
        # Create output directories
        self.train_path.mkdir(parents=True, exist_ok=True)
        
        # Define musical symbol classes
        self.classes = [
            'c1', 'c2', 'd1', 'd2', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'a1', 'a2', 'b1', 'b2',
            'sharp', 'flat', 'natural', 'clef', 'bar', 'dot', 'chord',
            '1', '2', '4', '8', '16', '32'
        ]
        
        # Create class directories
        for class_name in self.classes:
            (self.train_path / class_name).mkdir(exist_ok=True)
    
    def process_images(self):
        """Process all images in the dataset"""
        print("🎵 Starting dataset processing...")
        
        image_files = list(self.dataset_path.glob("*.jpg"))
        print(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        
        for i, image_file in enumerate(image_files[:50]):  # Process first 50 for testing
            print(f"Processing {image_file.name} ({i+1}/{len(image_files)})")
            
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"⚠️  Could not read {image_file.name}")
                    continue
                
                # Extract regions of interest (ROIs)
                rois = self.extract_rois(image)
                
                # Process each ROI
                for j, roi in enumerate(rois):
                    if self.is_valid_symbol(roi):
                        # Classify and save the symbol
                        class_name = self.classify_symbol(roi)
                        if class_name and class_name in self.classes:
                            output_path = self.train_path / class_name / f"{image_file.stem}_{j}.jpg"
                            cv2.imwrite(str(output_path), roi)
                            processed_count += 1
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
        
        print(f"✅ Processing complete! Extracted {processed_count} symbols")
        self.generate_dataset_info()
    
    def extract_rois(self, image):
        """Extract regions of interest from the image"""
        rois = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (adjust these thresholds)
            if 20 < w < 200 and 20 < h < 200:
                # Extract the region
                roi = image[y:y+h, x:x+w]
                
                # Resize to standard size
                roi = cv2.resize(roi, (100, 100))
                rois.append(roi)
        
        return rois
    
    def is_valid_symbol(self, roi):
        """Check if the ROI is a valid musical symbol"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check if there's enough contrast
        std = np.std(gray)
        if std < 10:  # Too uniform, likely background
            return False
        
        # Check if there's enough white pixels (symbol pixels)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        
        if white_pixels / total_pixels < 0.05:  # Less than 5% white pixels
            return False
        
        return True
    
    def classify_symbol(self, roi):
        """Classify the symbol (simplified classification)"""
        # This is a simplified classification
        # In a real system, you'd use a trained ML model
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Simple shape analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Analyze contour properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Simple classification based on area and shape
        if area < 500:
            return 'dot'
        elif area > 2000:
            return 'clef'
        else:
            # For demonstration, randomly assign to available classes
            import random
            return random.choice(['sharp', 'flat', 'natural', 'note'])
    
    def generate_dataset_info(self):
        """Generate dataset information"""
        dataset_info = {
            "name": "Mozart OMR Dataset",
            "version": "1.0",
            "description": "Musical symbol dataset for Optical Music Recognition",
            "classes": self.classes,
            "total_classes": len(self.classes),
            "created": "2024-11-30"
        }
        
        # Count images per class
        class_counts = {}
        for class_name in self.classes:
            class_dir = self.train_path / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))
                class_counts[class_name] = len(images)
        
        dataset_info["class_distribution"] = class_counts
        dataset_info["total_images"] = sum(class_counts.values())
        
        # Save dataset info
        info_path = self.output_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📊 Dataset info saved to {info_path}")
        print(f"📈 Total images in dataset: {dataset_info['total_images']}")
        print(f"📁 Classes: {dataset_info['total_classes']}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python setup_dataset.py <dataset_path> <output_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    
    processor = DatasetProcessor(dataset_path, output_path)
    processor.process_images()