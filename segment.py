from transformers import SamModel, SamProcessor
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and apply dynamic quantization
model = SamModel.from_pretrained("facebook/sam-vit-huge")
model = model.to(device)
# model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Load the image
img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[450, 600]]]


with torch.no_grad():
    inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
    outputs = model(**inputs)


# Process the masks and scores
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

print(scores)

print(masks[0].shape)

quit()
# masks.save("mask.png")


# Convert the mask to numpy array (if needed)
mask = masks[0].numpy()  # assuming there is only one mask

print(mask.shape)

quit()
mask = np.transpose(mask, (1, 2, 0))

print(mask)

quit()
# binary_mask = (mask > 0.5).astype(np.uint8)
# binary_mask = np.transpose(mask, (1, 2, 0)).astype(np.uint8)


# Plotting the mask using matplotlib
plt.figure(figsize=(10, 10))

# If you want to overlay the mask on the original image:
raw_image_np = np.array(raw_image)  # Convert raw image to numpy
plt.imshow(raw_image_np)  # Display the original image
plt.imshow(binary_mask, alpha=0.5, cmap="jet")  # Overlay the mask with transparency
plt.axis('off')  # Hide axes

# Save the figure
plt.savefig("output/mask.png")
# plt.show()
