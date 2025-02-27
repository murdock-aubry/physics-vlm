from transformers import pipeline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
from torchvision.ops import batched_nms

points_per_batch = 4

generator =  pipeline("mask-generation", device = 0, points_per_batch = points_per_batch)
image_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

outputs = generator(image_url, points_per_batch = points_per_batch)


scores = outputs["scores"] # length of scores = number of mask components
masks = outputs["masks"] 


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
plt.imshow(np.array(raw_image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
plt.savefig("output/masked_image.png")