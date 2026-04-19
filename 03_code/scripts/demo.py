import os
import argparse
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
import gradio as gr
import cv2
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.efficientnet_detector import EfficientNetDetector

def load_detector(checkpoint_path, device):
    model = EfficientNetDetector()
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("WARNING: Checkpoint not found. Initialized with random weights.")
    model.to(device)
    model.eval()
    return model

def extract_and_predict(video_path, model, mtcnn, device):
    if video_path is None:
        return "No video uploaded.", []

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Extract 16 evenly spaced frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, length // 16)
    
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened() and extracted_count < 16:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            extracted_count += 1
        frame_count += 1
    cap.release()

    face_crops = []
    predictions = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for img in frames:
        # Detect and crop single face per frame
        face_tensor = mtcnn(img)
        if face_tensor is not None:
            # MTCNN returns normalized array. We actually want PIL image to display
            # To get PIL crop for display and tensor for model:
            boxes, probs = mtcnn.detect(img)
            if boxes is not None:
                box = boxes[0]
                face = img.crop((box[0], box[1], box[2], box[3]))
                face_crops.append(face)
                
                # Transform for model
                model_input = transform(face).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(model_input)
                    pred = output["prediction"].item()
                    predictions.append(pred)

    if not predictions:
        return "No faces detected in the video.", []

    avg_pred = sum(predictions) / len(predictions)
    
    verdict = "REAL" if avg_pred < 0.5 else "FAKE"
    confidence = (1 - avg_pred) if verdict == "REAL" else avg_pred
    
    final_output = f"Verdict: {verdict}\nConfidence: {confidence:.2%}"
    
    return final_output, face_crops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="05_results/models/best_efficientnet.pth", help="Path to best model pth")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_detector(args.checkpoint, device)
    mtcnn = MTCNN(image_size=224, margin=40, keep_all=False, device=device)

    def predict_interface(video):
        return extract_and_predict(video, model, mtcnn, device)

    app = gr.Interface(
        fn=predict_interface,
        inputs=gr.Video(),
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.Gallery(label="Extracted Faces Detected")
        ],
        title="Robust Deepfake Detector",
        description="Upload a video to analyze it for deepfake alterations. This model utilizes spatial classification via EfficientNet. Adversarial features support will be integrated soon."
    )
    
    # allow file sharing for colab, but for EC2 server share=True is required
    app.launch(share=True)

if __name__ == "__main__":
    main()
