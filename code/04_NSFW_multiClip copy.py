from PIL import Image
import requests 
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
from PIL import Image, ImageDraw, ImageFont


# List of emotional keywords
NSFW_keywords = ['sexual', 
            'nude', 
            'sex', 
            '18+', 
            'naked', 
            'nsfw', 
            'porn', 
            'dick', 
            'vagina', 
            'naked person (approximation)',
            'explicit content', 
            'uncensored', 
            'fuck', 
            'nipples', 
            'nipples (approximation)', 
            'naked breasts', 
            'areola']
i = 0
while True:

    url = 'https://thispersondoesnotexist.com/'
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    #inputKey = input("Enter the URL of the image: ")
    #url = inputKey
    #img = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=NSFW_keywords, images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
  
    # Get the index of the NSFW keyword with the highest probability
    index_of_highest_prob_NSFW = probs.argmax().item()

    # Get the NSFW keyword with the highest probability
    highest_prob_NSFW = NSFW_keywords[index_of_highest_prob_NSFW]

    # Get the corresponding probability
    highest_prob = probs[0, index_of_highest_prob_NSFW].item()

    print(f"Highest probability NSFW: {highest_prob_NSFW}\tProbability: {highest_prob:.2f}")


    # Overlay scores and labels on the camera image
    draw = ImageDraw.Draw(img)
    font_size = 80  # Increased font size even more
    font = ImageFont.truetype("NotoSans-Regular.ttf", font_size)
   
    draw.text((10, 50), f"{highest_prob_NSFW}",stroke=(0,0,0),fill=(255, 255, 255), font=font)
    i += 1

    #print(f"{i}\tscore: {now:.2f}")
    for j, NSFW in enumerate(NSFW_keywords):
        prob_NSFW = probs[0, j].item()
        print(f"\NSFW: {NSFW}\tProbability: {prob_NSFW:.2f}")


    img.save(f'NSFW/this_{i}.jpg')
    i += 1
