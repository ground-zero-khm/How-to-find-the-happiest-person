import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
from torch.nn import functional as F
from clip import clip

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

# Load the ResNet50 model for image preprocessing
resnet_model = resnet50(pretrained=True).to(device)
resnet_model.eval()
preprocess_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the target class
target_class = "happy"

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the OpenCV frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image for CLIP
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)

    # Preprocess the image for ResNet50
    image_resnet = preprocess_resnet(image).unsqueeze(0).to(device)

    # Run the image through ResNet50
    with torch.no_grad():
        features = resnet_model(image_resnet)
        features = F.normalize(features, dim=1)

    # Generate text features from the image using CLIP
    with torch.no_grad():
        image_features = model.encode_image(image_preprocessed)
        image_features = F.normalize(image_features, dim=1)

        text = clip.tokenize([target_class]).to(device)
        text_features = model.encode_text(text)
        text_features = F.normalize(text_features, dim=1)

        # Calculate the similarity score between the image and the target class
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Check if the person is happy based on the similarity score
    is_happy = similarity.item() > 0.5

    # Display the result on the frame
    cv2.putText(frame, f"Happy: {is_happy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Camera', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
