import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from huggingface_hub import hf_hub_download
from utils import *


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
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






quit()


# Load the model in half-precision
model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16).to(device)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

# Load the video as an np.arrau, sampling uniformly 8 frames
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)
total_frames = container.streams.video[0].frames

indices = np.arange(0, total_frames, total_frames / 8).astype(int)
video = read_video_pyav(container, indices)

# For better results, we recommend to prompt the model in the following format
prompt = "USER: <video>\nWhy is this funny? ASSISTANT:"
inputs = processor(text=prompt, videos=video, return_tensors="pt").to(device)

output = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)
decoded_out = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(decoded_out)
