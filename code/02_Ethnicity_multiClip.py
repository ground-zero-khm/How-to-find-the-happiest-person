from PIL import Image
import requests 
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# List of emotional keywords
ethnicty_keywords = [
    "African",
    "Caribbean",
    "Indian",
    "Melanesian",
    "Australasian/Aboriginal",
    "Chinese",
    "Japanese",
    "Korean",
    "Polynesian",
    "European",
    "Latin American",
    "Arabic",
]
i = 0


while True:


    #url = 'https://thispersondoesnotexist.com/'
    #img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    #inputKey = input("Enter the URL of the image: ")
    #url = inputKey
    #img = Image.open(requests.get(url, stream=True).raw)


    img = Image.open("schnitzel.png")

    inputs = processor(text=ethnicty_keywords, images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
  
    # Get the index of the ethnicity keyword with the highest probability
    index_of_highest_prob_ethnicity = probs.argmax().item()

    # Get the ethnicity keyword with the highest probability
    highest_prob_ethnicity = ethnicty_keywords[index_of_highest_prob_ethnicity]

    # Get the corresponding probability
    highest_prob = probs[0, index_of_highest_prob_ethnicity].item()

    print(f"Highest probability ethnicity: {highest_prob_ethnicity}\tProbability: {highest_prob:.2f}")

    #print(f"{i}\tscore: {now:.2f}")
    for j, ethnicity in enumerate(ethnicty_keywords):
        prob_ethnicity = probs[0, j].item()
        print(f"ethnicity: {ethnicity}\tProbability: {prob_ethnicity:.2f}")


    img.save(f'ethnicity/this_{i}.jpg')
    i += 1
