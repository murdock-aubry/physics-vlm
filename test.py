from PIL import Image
import requests
import numpy as np
import random 
import torch
import av
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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



base_model_name = "LanguageBind/Video-LLaVA-7B-hf"
adapter_name = "murdockaubry/Video-LLaVA-7B-natural"

base_model = VideoLlavaForConditionalGeneration.from_pretrained(base_model_name, load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, adapter_name)


processor = VideoLlavaProcessor.from_pretrained(base_model_name)
processor.patch_size = 14
processor.vision_feature_select_strategy = "uniform"

prompt = "USER: <video>\nBriefly, why is this video funny? ASSISTANT:"

video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

inputs = processor(text=prompt, videos=clip, return_tensors="pt")

generate_ids = model.generate(
    **inputs, 
    max_new_tokens=20,
    do_sample=False  # Use greedy decoding
)

print(processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0], flush = True)
