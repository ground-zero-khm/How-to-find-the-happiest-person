from PIL import Image
import requests 
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# List of emotional keywords
emotional_keywords = [
    "a photo of happiness",
    "a photo of sadness",
    "a photo of surprise",
    "a photo of anger",
    "a photo of fear",
    "a photo of disgust",
]

i = 0
while True:
    inputKey = input("Enter the URL of the image: ")
    url = inputKey
    img = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=emotional_keywords, images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
  
    # Get the index of the emotional keyword with the highest probability
    index_of_highest_prob_emotion = probs.argmax().item()

    # Get the emotional keyword with the highest probability
    highest_prob_emotion = emotional_keywords[index_of_highest_prob_emotion]

    # Get the corresponding probability
    highest_prob = probs[0, index_of_highest_prob_emotion].item()

    print(f"Highest probability emotion: {highest_prob_emotion}\tProbability: {highest_prob:.2f}")
    img.save(f'outputimage/this_{i}.jpg')
    i += 1
