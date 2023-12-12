from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from clip_interrogator import Config, Interrogator
import requests

url = 'https://thispersondoesnotexist.com/'
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
print(ci.interrogate(img))

