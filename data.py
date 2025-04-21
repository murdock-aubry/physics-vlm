from PIL import Image
import requests
import numpy as np
import random 
import torch
import av
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


num_frames = 16

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

split = "continuity"
data_path = f"/projects/dynamics/vlm-tmp/data/inflevel_lab/{split}/"
video_files = [f for f in os.listdir(data_path) if f.endswith('.mp4')]

base_model_name = "LanguageBind/Video-LLaVA-7B-hf"
adapter_name = "mvrdock/Video-LLaVA-7B-natural"

base_model = VideoLlavaForConditionalGeneration.from_pretrained(base_model_name, load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, adapter_name)

processor = VideoLlavaProcessor.from_pretrained(base_model_name)
processor.patch_size = 14
processor.vision_feature_select_strategy = "uniform"





for ivideo, path in enumerate(video_files):
    # Extract the portion between last two underscores
    parts = path.split('_')

    part = parts[-3]  # Get the third to last part to get label

    # Check if all characters in vv_part are the same
    if part[0] == part[1]:
        label = "Yes"
    else:
        label = "No"

    # get and process video
    video_path = data_path + path
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(container, indices)

    prompt = "USER: <video>\nIs this video physically plausible? Answer only 'yes' or 'no'. ASSISTANT:"

    # Process both videos for input
    inputs = processor(
        text=prompt,
        videos=clip,  # Pass both videos as a list
        return_tensors="pt"
    )

    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=10,
        do_sample=False  # Use greedy decoding
    )

    # Get the full generated text
    full_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    # Extract only the assistant's response by finding the last occurrence of "ASSISTANT:"
    response = full_text.split("ASSISTANT:")[-1].strip().replace("</s>", "")
    print(response, flush=True)
    print(response == label)
