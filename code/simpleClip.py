from PIL import Image
import requests 
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print('load model')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print('load processor')

i = 0
while i == 0:
    

    #inputKey = input()

    ####################input website#############################
    #url = inputKey
    #img = Image.open(requests.get(url, stream=True).raw)


    ##########this person does not exist##########################
    #url = 'https://thispersondoesnotexist.com/'
    #img = Image.open(requests.get(url, stream=True).raw).convert("RGB")


    ##########load image from dist##############################
    img = Image.open("beach.jpg")

    

    inputs = processor(text=["AI"], images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    #print(probs.item())
    now = logits_per_image.item()
   
    prob_happy = probs[0,:].item()
    print(f"image Number 1 : {i}\tscore:{now}\tprob:{prob_happy}")




    i = i + 1