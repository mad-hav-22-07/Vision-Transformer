"""Generate dummy images and masks for testing the training pipeline."""
import os
import cv2
import numpy as np
from pathlib import Path

def create_dummy_data(base_dir="dataset", num_train=10, num_val=4):
    base = Path(base_dir)
    
    dirs = [
        base / "images" / "train",
        base / "images" / "val",
        base / "masks" / "train",
        base / "masks" / "val"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        
    def generate_set(folder_type, num_images):
        img_dir = base / "images" / folder_type
        mask_dir = base / "masks" / folder_type
        
        for i in range(num_images):
            # 1. Create a random noisy image (360x640 RGB)
            # Use some colored noise just so it's not completely black
            img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"dummy_{i:03d}.jpg"), img)
            
            # 2. Create a random mask with classes 0, 1, 2
            # Mostly background (0), some dashed (1), some solid (2)
            mask = np.zeros((360, 640), dtype=np.uint8)
            
            # Draw a thick random line for dashed (class 1)
            cv2.line(mask, (100, 360), (300, 100), 1, thickness=15)
            
            # Draw a thick random line for solid (class 2)
            cv2.line(mask, (500, 360), (400, 100), 2, thickness=15)
            
            cv2.imwrite(str(mask_dir / f"dummy_{i:03d}.png"), mask)
            
    print(f"Generating {num_train} training samples...")
    generate_set("train", num_train)
    
    print(f"Generating {num_val} validation samples...")
    generate_set("val", num_val)
    
    print("Dummy dataset creation complete!")

if __name__ == "__main__":
    create_dummy_data()
