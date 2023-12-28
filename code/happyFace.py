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
emotional_keywords = [
    "a photo of happiness",
    "a photo of sadness",
    "a photo of surprise",
    "a photo of anger",
    "a photo of fear",
    #"a photo of disgust",
    "a photo of excitement",
    "a photo of contentment",
    "a photo of love",
    "a photo of anxiety",
    "a photo of jealousy",
    "a photo of hope",
    "a photo of boredom",
    "a photo of pride",
    "a photo of shame",
    "a photo of gratitude",
    "a photo of sympathy",
    "a photo of compassion",
    "a photo of curiosity",
    "a photo of awe",
    "a photo of guilt",
    "a photo of regret",
    "a photo of amusement",
    "a photo of irritation",
    "a photo of surprise",
]



import json
with open('countries.json', 'r') as f:
    data = json.load(f)

countries = []
for item in data:
    countries.append("person from the country "+item['name'])


emotional_keywords = countries

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
    inputs = processor(text=emotional_keywords, images=img, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

    # Get the index of the emotional keyword with the highest probability
    index_of_highest_prob_emotion = probs.argmax().item()

    # Get the emotional keyword with the highest probability
    highest_prob_emotion = emotional_keywords[index_of_highest_prob_emotion]

    # Get the corresponding probability
    highest_prob = probs[0, index_of_highest_prob_emotion].item()

    print(f"Highest probability emotion: {highest_prob_emotion}\tProbability: {highest_prob:.2f}")


    #print(f"{i}\tscore: {now:.2f}")
    for j, emotion in enumerate(emotional_keywords):
        prob_emotion = probs[0, j].item()
        print(f"\tEmotion: {emotion}\tProbability: {prob_emotion:.2f}")

    # Overlay scores and labels on the camera image
    draw = ImageDraw.Draw(img)
    font_size = 80  # Increased font size even more
    font = ImageFont.truetype("NotoSans-Regular.ttf", font_size)
   
    draw.text((10, 50), f"{highest_prob_emotion}",stroke=(0,0,0),fill=(255, 255, 255), font=font)
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
