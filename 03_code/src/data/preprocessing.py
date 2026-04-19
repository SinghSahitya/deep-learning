"""
Preprocessing pipeline: video -> frames -> face crops -> balanced splits.

Owner: Sahitya

Functions:
    extract_frames_in_memory: Read frames from a video into memory (PIL Images)
    crop_faces_batch:         Detect + crop faces for a batch of frames using MTCNN
    extract_frames:           Extract every Nth frame from a video and save as PNG
    crop_faces:               Detect + crop a face from a single frame using MTCNN
    balance_dataset:          Undersample majority class for balanced training
    create_splits:            Stratified train/val/test split, saved as CSVs
"""

import os
import random

import cv2
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def extract_frames_in_memory(video_path, frame_interval=10):
    """
    Extract every Nth frame from a video and return them as a list of
    (frame_count, PIL_Image) tuples, without saving to disk.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append((frame_count, img))

        frame_count += 1

    cap.release()
    return frames


def crop_faces_batch(frames_tuple, video_name, output_dir, detector):
    """
    Detect and crop faces for a batch of frames using MTCNN.

    Args:
        frames_tuple: list of (frame_count, PIL_Image)
        video_name: base name used in output filenames
        output_dir: directory to save cropped face PNGs
        detector: facenet_pytorch.MTCNN instance (with save_path support)

    Returns:
        list of saved face image paths (only for frames where a face was found)
    """
    if not frames_tuple:
        return []

    os.makedirs(output_dir, exist_ok=True)

    frame_counts = [f[0] for f in frames_tuple]
    images = [f[1] for f in frames_tuple]

    output_paths = [
        os.path.join(output_dir, f"{video_name}_frame_{c}.png") for c in frame_counts
    ]

    saved_paths = []
    try:
        faces = detector(images, save_path=output_paths)
        if faces is not None:
            for face_tensor, out_path in zip(faces, output_paths):
                if face_tensor is not None and os.path.exists(out_path):
                    saved_paths.append(out_path)
    except Exception as e:
        print(f"Error in batched cropping for {video_name}: {e}")

    return saved_paths


def extract_frames(video_path, output_dir, frame_interval=10):
    """
    Extract every Nth frame from a video and save as PNG files.

    Returns:
        list of saved frame paths
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    saved_paths = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            out_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.png")
            cv2.imwrite(out_path, frame)
            saved_paths.append(out_path)

        frame_count += 1

    cap.release()
    return saved_paths


def crop_faces(frame_path, output_path, detector):
    """
    Detect and crop the face from a single frame using MTCNN.

    Args:
        frame_path: path to input frame image
        output_path: where to save the cropped face
        detector: facenet_pytorch.MTCNN instance

    Returns:
        True if face was found and saved, False otherwise
    """
    try:
        img = Image.open(frame_path).convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_cropped = detector(img, save_path=output_path)
        return img_cropped is not None
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return False


def balance_dataset(real_paths, fake_paths, seed=42):
    """
    Balance dataset by undersampling the majority class.

    Returns:
        (balanced_real_paths, balanced_fake_paths) — both have same length
    """
    min_count = min(len(real_paths), len(fake_paths))

    random.seed(seed)
    balanced_real = random.sample(real_paths, min_count)
    balanced_fake = random.sample(fake_paths, min_count)

    return balanced_real, balanced_fake


def create_splits(
    image_paths,
    labels,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Stratified train/val/test split, saves CSV files with columns: path, label.

    Returns:
        (train_count, val_count, test_count)
    """
    df = pd.DataFrame({"path": image_paths, "label": labels})

    train_df, val_test_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=relative_test_ratio,
        stratify=val_test_df["label"],
        random_state=seed,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    return len(train_df), len(val_df), len(test_df)
