from PIL import Image
import requests 
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

i = 0
while True:
    
    inputKey = input()
    url = inputKey
    #img = Image.open(requests.get(url, stream=True).raw)


    url = 'https://thispersondoesnotexist.com/'
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")


    inputs = processor(text=["happy"], images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    #print(probs.item())
    now = logits_per_image.item()
   
    prob_happy = probs[0,:].item()
    print(f"{i}\tscore:{now}\tprob:{prob_happy}")
    img.save('outputimage/this.jpg')
    i = i + 1