from PIL import Image
import requests
import numpy as np
import random 
import torch
import av
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import math

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
import os
os.environ["HF_HOME"] = "/w/340/abarroso/huggingface"

from decord import VideoReader
import numpy as np
def save_frames_to_mp4(frames, output_path, fps=24):
    if len(frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    container = av.open(output_path, mode='w')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()

def random_resized_crop(images, target_height, target_width):
    buffer = torch.tensor(images, dtype=torch.float32)  # (T, H, W, C)
    buffer = buffer.permute(0, 3, 1, 2)  # (T, C, H, W)
    buffer = torch.nn.functional.interpolate(
        buffer,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False,
    )
    return buffer.permute(0, 2, 3, 1).numpy()  # back to (T, H, W, C)
    
def process_local_mp4(file_path, n=40, start_frame=330, max_frames=None):
    """
    Read every nth frame from a local MP4 file using Decord VideoReader.
    
    Args:
        file_path (str): Path to the local MP4 file
        n (int): Extract every nth frame
        start_frame (int): Frame index to start extraction from
        max_frames (int, optional): Maximum number of frames to extract
                                   (None means no limit)
        
    Returns:
        numpy.ndarray: Stack of video frames in RGB format
    """
    
    # Initialize VideoReader
    vr = VideoReader(file_path)
    frame_indices = range(start_frame, len(vr), n)
    
    if max_frames is not None:
        frame_indices = list(frame_indices)[:max_frames]
    # Decode frames
    frames_batch = vr.get_batch(frame_indices).asnumpy()
    return frames_batch.astype(np.uint8)



split = "continuity"
data_path = f"../Datasets/inflevel_lab/{split}/"


video = "center__continuity__greencup__bluestar__vi__RL.mp4"
video_frames = process_local_mp4(data_path + video)
#video2 = "center__continuity__greencup__bluestar__vV__RL.mp4"
#video2_frames = process_local_mp4s(data_path + video2)
#save_frames_to_mp4(video_frames, "output3.mp4", fps=1)
#save_frames_to_mp4(video2_frames, "output4.mp4", fps=1)
#videos = [video2_frames, video_frames]
# video2 = "left__continuity__orangecup__pepper__ii__RL.mp4"
# video2_frames = process_local_mp4(data_path + video2)

base_model_name = "LanguageBind/Video-LLaVA-7B-hf"
adapter_name = "mvrdock/Video-LLaVA-7B-natural"

base_model = VideoLlavaForConditionalGeneration.from_pretrained(base_model_name, load_in_4bit=True, cache_dir="/w/340/abarroso/huggingface/models")
model = PeftModel.from_pretrained(base_model, adapter_name)
prompt = "USER: <video>\nGive this video a physical feasibility score from 1 to 10, where 10 means it fully obeys the laws of physics and has no visual continuity errors (like objects disappearing or jumping positions). Respond with only a number. ASSISTANT:"
processor = VideoLlavaProcessor.from_pretrained(base_model_name)
processor.patch_size = 14
processor.vision_feature_select_strategy = "uniform"

# prompt = (
#     "USER: <video> <video>\n"
#     "Here are two videos. Which one violates physics more or are they the same video and why.\nASSISTANT:"
# )

# Process both videos for input
inputs = processor(
    text=prompt,
    videos=video_frames,  # Pass both videos as a list
    return_tensors="pt"
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)} (not a tensor)")
generate_ids = model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=False  # Use greedy decoding
)
response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
print(output, flush = True)
