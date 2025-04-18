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


def process_local_mp4(file_path, n=5, start_frame=0, max_frames=None):
    """
    Read every nth frame from a local MP4 file.
    
    Args:
        file_path (str): Path to the local MP4 file
        n (int): Extract every nth frame
        start_frame (int): Frame index to start extraction from
        max_frames (int, optional): Maximum number of frames to extract
                                   (None means no limit)
        
    Returns:
        numpy.ndarray: Stack of video frames in RGB format
    """
    frames = []
    count = 0
    
    # Open the local file
    with av.open(file_path) as container:
        # Get video stream
        stream = container.streams.video[0]
        
        # Decode frames
        for i, frame in enumerate(container.decode(video=0)):
            if i < start_frame:
                continue
                
            if (i - start_frame) % n == 0:
                frames.append(frame)
                count += 1
                
            if max_frames is not None and count >= max_frames:
                break
    
    # Convert frames to numpy arrays and stack them
    if not frames:
        return np.array([])  # Return empty array if no frames were collected
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


split = "continuity"
data_path = f"/projects/dynamics/vlm-tmp/data/inflevel_lab/{split}/"


video = "left__continuity__orangecup__pepper__ii__LR.mp4"
video_frames = process_local_mp4(data_path + video)

# video2 = "left__continuity__orangecup__pepper__ii__RL.mp4"
# video2_frames = process_local_mp4(data_path + video2)

base_model_name = "LanguageBind/Video-LLaVA-7B-hf"
adapter_name = "mvrdock/Video-LLaVA-7B-natural"

base_model = VideoLlavaForConditionalGeneration.from_pretrained(base_model_name, load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, adapter_name)

processor = VideoLlavaProcessor.from_pretrained(base_model_name)
processor.patch_size = 14
processor.vision_feature_select_strategy = "uniform"

prompt = "USER: <video>\nIs this video physically plausible? Answer only 'yes' or 'no'. ASSISTANT:"

# Process both videos for input
inputs = processor(
    text=prompt,
    videos=video_frames,  # Pass both videos as a list
    return_tensors="pt"
)

generate_ids = model.generate(
    **inputs, 
    max_new_tokens=10,
    do_sample=False  # Use greedy decoding
)

print(processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0], flush = True)
