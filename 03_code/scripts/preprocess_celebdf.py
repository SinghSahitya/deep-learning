import os
import argparse
import glob
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

# Add the src package to python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocessing import extract_frames, crop_faces, balance_dataset, create_splits

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Celeb-DF dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw dataset (04_data/celeb_df_raw)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output frames (04_data/celeb_df_frames)")
    parser.add_argument("--frame_interval", type=int, default=10, help="Extract 1 frame every N frames")
    return parser.parse_args()

def process_videos(video_paths, label_name, output_base_dir, frame_interval, detector):
    """
    Extract frames and crop faces for a list of videos.
    Returns a list of saved face image paths.
    """
    face_paths = []
    
    # We save faces inside output_base_dir / label_name
    faces_dir = os.path.join(output_base_dir, label_name)
    os.makedirs(faces_dir, exist_ok=True)
    
    # Temporary directory for raw frames
    temp_frames_dir = os.path.join(output_base_dir, f"temp_{label_name}")
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    print(f"Processing {len(video_paths)} videos for {label_name}...")
    for video_path in tqdm(video_paths, desc=f"Extracting {label_name}"):
        
        # 1. Extract frames
        raw_frames = extract_frames(video_path, temp_frames_dir, frame_interval)
        
        # 2. Crop faces
        for frame_path in raw_frames:
            # Output path for the face
            filename = os.path.basename(frame_path)
            face_path = os.path.join(faces_dir, filename)
            
            # Crop using MTCNN
            success = crop_faces(frame_path, face_path, detector)
            if success:
                face_paths.append(face_path)
                
            # Clean up the raw frame to save space
            os.remove(frame_path)
            
    # Remove temp dir
    os.rmdir(temp_frames_dir)
    return face_paths

def main():
    args = parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize MTCNN
    # image_size=224, margin=40 equivalent to ~20% padding
    detector = MTCNN(image_size=224, margin=40, keep_all=False, device=device)
    
    # Define input paths
    real_video_dir = os.path.join(args.input_dir, "Celeb-real")
    fake_video_dir = os.path.join(args.input_dir, "Celeb-synthesis")
    
    # You could also add YouTube-real if you wanted, but following the task file, we'll stick to Celeb-real for now.
    
    real_videos = glob.glob(os.path.join(real_video_dir, "*.mp4"))
    fake_videos = glob.glob(os.path.join(fake_video_dir, "*.mp4"))
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos.")
    
    # Process Real Videos
    real_face_paths = process_videos(real_videos, "real", args.output_dir, args.frame_interval, detector)
    
    # Process Fake Videos
    fake_face_paths = process_videos(fake_videos, "fake", args.output_dir, args.frame_interval, detector)
    
    print(f"Total extracted faces - Real: {len(real_face_paths)}, Fake: {len(fake_face_paths)}")
    
    # Balance dataset
    print("Balancing dataset...")
    balanced_real, balanced_fake = balance_dataset(real_face_paths, fake_face_paths)
    print(f"Balanced counts - Real: {len(balanced_real)}, Fake: {len(balanced_fake)}")
    
    # Create splits
    print("Creating train/val/test splits...")
    all_paths = balanced_real + balanced_fake
    # 0 = real, 1 = fake
    all_labels = [0] * len(balanced_real) + [1] * len(balanced_fake)
    
    # The image paths are absolute or relative to where the script is run.
    # It's better to store relative paths in the CSV.
    # We will use absolute paths for ease of loading later.
    all_paths = [os.path.abspath(p) for p in all_paths]
    
    train_count, val_count, test_count = create_splits(all_paths, all_labels, args.output_dir)
    print(f"Splits created successfully!")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")
    print(f"CSV files saved to {args.output_dir}")

if __name__ == "__main__":
    main()
