import numpy as np
import random 
import torch
import av
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import io
import cv2
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import re
import json

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
    
def collect_camera_folders(root):
    return [
        os.path.join(root, folder, str(cam_id))
        for folder in os.listdir(root)
        if os.path.isdir(os.path.join(root, folder))
        for cam_id in range(1, 5)
        if os.path.exists(os.path.join(root, folder, str(cam_id), "scene")) and
           os.path.exists(os.path.join(root, folder, str(cam_id), "status.json"))
    ]


def encode_frames_to_memory(frames, fps=24):
    if len(frames) == 0:
        return b""

    height, width, _ = frames[0].shape
    output = io.BytesIO()

    container = av.open(output, mode='w', format='mp4')
    stream = container.add_stream('libx264', rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'

    for frame_array in frames:
        frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    output.seek(0)

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

def load_scene_frames_parallel(folder_path, skip=15, ext=".png", max_workers=8):
    folder_path = os.path.join(folder_path, "scene")
    pattern = re.compile(r"scene_(\d+)\{}".format(ext))
    
    # List and sort files by extracted number
    files = [
        f for f in os.listdir(folder_path)
        if pattern.match(f)
    ]
    files.sort(key=lambda f: int(pattern.match(f).group(1)))

    # Select every `skip` frames (1-indexed logic: 1, 6, 11, ...)
    selected_files = files[::skip]

    def load_image(filename):
        path = os.path.join(folder_path, filename)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = list(executor.map(load_image, selected_files))

    return np.stack(frames, axis=0).astype(np.uint8)

def get_is_possible_from_json(folder_path):
    # Define the file path assuming it's in the current directory
    file_path = os.path.join(folder_path, "status.json")
    
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Access the 'is_possible' field from the JSON structure
    return data.get("header", {}).get("is_possible", None)

def round_softmax(probs, decimal_places=4):
    # Step 1: Round each probability to the desired decimal places
    rounded = torch.round(probs * (10**decimal_places)) / (10**decimal_places)
    
    # Step 2: Normalize to ensure they sum to 1
    normalized = rounded / rounded.sum()
    
    return normalized

def get_yes_no_probability(model, inputs, processor):
    # Get the generated token IDs and the first new token
    generate_ids = model.generate(
    **inputs, 
    max_new_tokens=10,
    do_sample=False,  # Use greedy decoding
    output_scores=True,
    return_dict_in_generate=True)
    generated_token_ids = generate_ids["sequences"][0]
    generated_token_ids = generated_token_ids[len(inputs["input_ids"][0]):]
    
    # Get the scores (logits) for each generated token
    scores = generate_ids["scores"]
    
    if generated_token_ids.numel() == 0 or len(scores) == 0:
        raise ValueError("No generated tokens or scores found.")
    
    # Process only the first generated token
    token_logits = scores[0]

    # Get token IDs for "Yes" and "No"
    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    # Get logits for yes and no
    yes_logit = token_logits[0, yes_token_id]
    no_logit = token_logits[0, no_token_id]

    # Softmax just over those two logits
    yes_no_logits = torch.tensor([yes_logit, no_logit])
    probs = F.softmax(yes_no_logits, dim=0)
    normalized_probs = round_softmax(probs, decimal_places=4)
    yes_prob = normalized_probs[0].item()
    no_prob = normalized_probs[1].item()
    return {
        "yes_prob": yes_prob,
        "no_prob": no_prob,
    }

def normal_resp(model, inputs, processor):
    generate_ids = model.generate(
    **inputs, 
    max_new_tokens=10,
    do_sample=False,  # Use greedy decoding
    #output_scores=True,
    #return_dict_in_generate=True
    )
    response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    #response_ids = generate_ids["sequences"][:, inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output

def cot_resp(model,inputs, processor, llm_model,llm_processor):
    generate_ids = model.generate(
    **inputs, 
    max_new_tokens=200,
    do_sample=False,  # Use greedy decoding
    #output_scores=True,
    #return_dict_in_generate=True
)
    response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

    llava_output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(f"CoT Reasoning Output from LLaVA: {llava_output}", flush=True)
    llm_prompt = f"""
    ### Instruction:
    Answer with "yes" or "no" only. If the statement clearly indicates that **an object** skips or teleports, answer "no". If the statement suggests that **all objects** move without skipping or teleporting, answer "yes".
    Statment:
    {llava_output}
    ### Response:\n"""
    inputs_pythia = llm_processor(llm_prompt, return_tensors="pt").to(llm_model.device)
    pythia_generate_ids = llm_model.generate(
        **inputs_pythia, 
        max_new_tokens=2,  # Limit to a short answer like "yes" or "no"
        do_sample=False,  # Use greedy decoding
    )

    # Decode the Pythia output to get the summarized result

    pythia_resp_ids = pythia_generate_ids[:, inputs_pythia["input_ids"].shape[1]:]
    return filter_alphabetic(llm_processor.decode(pythia_resp_ids[0], skip_special_tokens=True))
    # Print the final summarized response from Pythia
def filter_alphabetic(text):
    # This regex will keep only alphabetic characters (both lowercase and uppercase) and spaces
    return re.sub(r'[^a-zA-Z ]+', '', text)
  
#for intphys    
base_folder = "/w/340/abarroso/Datasets/dev/O3/06/2"
frames_batch = load_scene_frames_parallel(base_folder, skip=7)
output_acc = get_is_possible_from_json(base_folder)
print(len(frames_batch))

#for inflevel
#split = "continuity"
#data_path = f"../Datasets/inflevel_lab/{split}/"
#video = "center__continuity__greencup__bluestar__vi__RL.mp4"
#video_path = os.path.join(data_path, video)
#frames_batch = process_local_mp4(video_path, 20)

base_model_name = "LanguageBind/Video-LLaVA-7B-hf"
adapter_name = "mvrdock/Video-LLaVA-7B-natural"

base_model = VideoLlavaForConditionalGeneration.from_pretrained(base_model_name, load_in_4bit=True, cache_dir="/w/340/abarroso/huggingface/models")
model = PeftModel.from_pretrained(base_model, adapter_name, cache_dir ="/w/340/abarroso/huggingface/models")

processor = VideoLlavaProcessor.from_pretrained(base_model_name)
llm_model_name = "vicgalle/gpt2-open-instruct-v1"
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir="/w/340/abarroso/huggingface/models")
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, cahce_dir="/w/340/abarroso/huggingface/models")
processor.patch_size = 14
processor.vision_feature_select_strategy = "uniform"
prompt = """User: <video>\n Do all objects in the video move without skipping or teleporting through space? Explain why or why not.\nAssistant:"""

inputs = processor(
    text=prompt,
    videos=frames_batch, 
    return_tensors="pt"
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

generate_ids = model.generate(
    **inputs, 
    max_new_tokens=5,
    do_sample=False,  # Use greedy decoding
)


print(f"Final Output: {cot_resp(model, inputs, processor, llm_model, llm_tokenizer)}",)
print(f"acc_output:{output_acc}", flush=True)
