from torch.utils.data import Dataset
import os
import numpy as np
import av
import torch 
import re
import cv2
from decord import VideoReader
from concurrent.futures import ThreadPoolExecutor
import json

class Inflevel_Dataset(Dataset):
    def __init__(self, video_dir, processor, prompt, experiment_fields,start_times, n=40, max_frames=None):
        self.video_dir = video_dir
        self.n = n
        self.start_times = start_times
        self.max_frames = max_frames
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4") and not f.startswith("._") and f in start_times] 
        self.processor = processor
        self.prompt = prompt
        self.fields = experiment_fields
        #all_start_frames = [v['start_frame'] for v in start_times.values()]
        #self.median_start_frame = int(np.median(all_start_frames))
    
    def __len__(self):
        return len(self.video_files)
    
    def process_local_mp4(self, file_path, start_frame):
        # Initialize VideoReader
        vr = VideoReader(file_path)
        frame_indices = range(start_frame, len(vr), self.n)
        
        #if max_frames is not None:
        #    frame_indices = list(frame_indices)[:max_frames]
        # Decode frames
        frames_batch = vr.get_batch(frame_indices).asnumpy()
        return frames_batch.astype(np.uint8)
        
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        start_frame = self.start_times[video_file]['start_frame']
        #start_frame = self.start_times.get(video_file, {}).get('start_frame', self.median_start_frame)
        video_frames = self.process_local_mp4(video_path, start_frame)

        # Remove the type (continuity or gravity)
        
        video_parts = video_file.split("__")
        video_parts[-1] = video_parts[-1].replace('.mp4', '')
        del video_parts[1]  # assumes type is always second

        prompt = self.prompt
        inputs = self.processor(
            text=prompt,
            videos=video_frames,
            return_tensors="pt"
        )

        if len(video_parts) == len(self.fields):
            metadata = {key: part for key, part in zip(self.fields, video_parts)}
            return inputs, video_file, *metadata.values()
        else:
            print("Mismatch in number of fields:", video_file)
            return None
        
def Inflevel_collate_fn_factory(fields):
    def collate_fn(batch):
        inputs = {
            key: torch.cat([example[0][key] for example in batch], dim=0)
            for key in batch[0][0]
        }
        video_files = [example[1] for example in batch]

        # Gather metadata per field
        metadata_dict = {field: [] for field in fields}
        for example in batch:
            for i, field in enumerate(fields):
                metadata_dict[field].append(example[2 + i])

        return inputs, video_files, metadata_dict
    return collate_fn

class IntPhysDataset(Dataset):
    def __init__(self, video_dir, processor, prompt, n=7):
        self.video_dir = video_dir
        self.n = n
        self.video_files = self.collect_camera_folders()
        self.processor = processor
        self.prompt = prompt
    
    def __len__(self):
        return len(self.video_files)
    
    
    def collect_camera_folders(self):
        return [
            os.path.join(self.video_dir, folder, str(cam_id))
            for folder in os.listdir(self.video_dir)
            if os.path.isdir(os.path.join(self.video_dir, folder))
            for cam_id in range(1, 5)
            if os.path.exists(os.path.join(self.video_dir, folder, str(cam_id), "scene")) and
            os.path.exists(os.path.join(self.video_dir, folder, str(cam_id), "status.json"))
        ]
    def load_scene_frames_parallel(self, folder_path, ext=".png", max_workers=8):
        folder_path = os.path.join(folder_path, "scene")
        pattern = re.compile(r"scene_(\d+)\{}".format(ext))
        
        # List and sort files by extracted number
        files = [
            f for f in os.listdir(folder_path)
            if pattern.match(f)
        ]
        files.sort(key=lambda f: int(pattern.match(f).group(1)))

        # Select every `skip` frames (1-indexed logic: 1, 6, 11, ...)
        selected_files = files[::self.n]

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

    def get_is_possible_from_json(self, folder_path):
        # Define the file path assuming it's in the current directory
        file_path = os.path.join(folder_path, "status.json")
        
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Access the 'is_possible' field from the JSON structure
        return data.get("header", {}).get("is_possible", None)
    
    def __getitem__(self, idx):
        video_folder = self.video_files[idx]
        label = self.get_is_possible_from_json(video_folder)
        video_frames = self.load_scene_frames_parallel(video_folder)
        prompt = self.prompt
        inputs = self.processor(
            text=prompt,
            videos=video_frames,
            return_tensors="pt"
        )
        relative_path = os.path.relpath(video_folder, self.video_dir)
        return inputs, relative_path, bool(label)
