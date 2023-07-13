import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the target classes
target_classes = ["happy", "sad"]

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    max_face_area = 0
    max_face_img = None

    for (x, y, w, h) in faces:
        # Select the face region
        face_img = frame[y:y+h, x:x+w]

        # Calculate the face area
        face_area = w * h

        if face_area > max_face_area:
            max_face_area = face_area
            max_face_img = face_img

    if max_face_img is not None:
        # Convert the OpenCV face image to a PIL image
        face_pil = Image.fromarray(cv2.cvtColor(max_face_img, cv2.COLOR_BGR2RGB))

        # Preprocess the face image using the CLIP processor
        inputs = processor(
            images=face_pil,
            texts=["dummy text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Forward pass through the CLIP model
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # This is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # Softmax to get the label probabilities

        # Determine the predicted emotion label
        max_prob, max_idx = probs[0].max(dim=0)
        predicted_emotion = target_classes[max_idx.item()]
        similarity_score = max_prob.item()

        # Draw rectangle around the face
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the predicted emotion and similarity score
        emotion_label = f"{predicted_emotion.capitalize()} ({similarity_score:.2f})"
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Camera', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
