from torch.utils.data import Dataset
import os
import numpy as np
import av
import torch 
from decord import VideoReader

class VideoDataset(Dataset):
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
        
def collate_fn_factory(fields):
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