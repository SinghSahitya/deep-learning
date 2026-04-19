import os
import cv2
from PIL import Image
import torch
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract every Nth frame from a video using OpenCV.
    - Open video with cv2.VideoCapture
    - Read frames, save every `frame_interval`-th frame as PNG
    - Name format: {video_name}_frame_{frame_number}.png
    - Return list of saved frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    
    saved_frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{frame_count}.png"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            saved_frames.append(output_path)
            
        frame_count += 1
        
    cap.release()
    return saved_frames

def crop_faces(frame_path, output_path, detector):
    """
    Detect and crop the face from a frame using MTCNN.
    - detector: facenet_pytorch.MTCNN instance
    - Detect face bounding box
    - Crop with some margin (MTCNN handles margin internally if set during initialization)
    - Resize to 224x224 (handled by MTCNN)
    - Save to output_path
    - Return True if face found, False otherwise
    """
    try:
        img = Image.open(frame_path).convert('RGB')
        # detector returns a tensor if a face is found, or None. 
        # By setting save_path, MTCNN will crop and save the image directly.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_cropped = detector(img, save_path=output_path)
        
        if img_cropped is not None:
            return True
        return False
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return False

def balance_dataset(real_paths, fake_paths):
    """
    Balance the dataset by undersampling the majority class.
    - Count real and fake samples
    - Randomly sample from the larger class to match the smaller
    - Return balanced lists
    """
    min_count = min(len(real_paths), len(fake_paths))
    
    # Shuffle and sample
    random.seed(42)
    balanced_real = random.sample(real_paths, min_count)
    balanced_fake = random.sample(fake_paths, min_count)
    
    return balanced_real, balanced_fake

def create_splits(image_paths, labels, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/val/test using sklearn.model_selection.train_test_split.
    - Stratify by label
    - Save split info as CSV files: train.csv, val.csv, test.csv
    - Each CSV has columns: path, label
    """
    df = pd.DataFrame({'path': image_paths, 'label': labels})
    
    # Split into train and (val + test)
    train_df, val_test_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df['label'], random_state=42
    )
    
    # Split (val + test) into val and test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        val_test_df, test_size=relative_test_ratio, stratify=val_test_df['label'], random_state=42
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    return len(train_df), len(val_df), len(test_df)
