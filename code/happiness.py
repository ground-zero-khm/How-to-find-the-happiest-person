import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel
import time
# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set the target text and initialize happiness and sadness scores
target_text = ["happy", "sad"]  # Modified target text

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Main loop to capture and process frames from the camera
i = 0  
happiness = 0
sadness = 0
while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the captured frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
  
    # Process each target text separately
    for idx, text in enumerate(target_text):
        # Preprocess the image using the CLIP processor
        inputs = processor(text=[text], images=img, return_tensors="pt", padding=True)

        # Perform forward pass through the CLIP model
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image

        now = logits_per_image.item()
        print(f"{i}\tText: {text}\tNow: {now:.2f}\t")

        if idx == 0:
            happiness = now
        elif idx == 1:
            sadness = now
        # Overlay scores and labels on the camera image
        draw = ImageDraw.Draw(img)
        font_size = 80  # Increased font size even more
        font = ImageFont.truetype("NotoSans-Regular.ttf", font_size)
        draw.text((10, 10 + 100 * idx), f"{text}: {now:.2f}", stroke=(0,0,0),fill=(255, 255, 255), font=font)
    
    if happiness > sadness:
        showText = "happy"
    else:
        showText = "sad"

    draw.text((10, 200), f"Target: {showText}",stroke=(0,0,0),fill=(255, 255, 255), font=font)
    i += 1

    # Convert the PIL Image back to OpenCV format for display
    frame_with_text = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Display the frame in a window
    cv2.imshow('Camera Image', frame_with_text)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.01)

# Release the VideoCapture and close the window
cap.release()
cv2.destroyAllWindows()
