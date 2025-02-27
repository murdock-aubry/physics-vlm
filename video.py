from transformers import pipeline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
import cv2
import os
from tqdm import tqdm

class VideoSegmentationPipeline:
    def __init__(self, device=0, points_per_batch=4, output_dir="output"):
        """
        Initialize the video segmentation pipeline.
        
        Args:
            device: Device to run the model on (0 for GPU, -1 for CPU)
            points_per_batch: Number of points to process in a batch
            output_dir: Directory to save output frames and video
        """
        self.points_per_batch = points_per_batch
        self.generator = pipeline("mask-generation", device=device, points_per_batch=points_per_batch)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def show_mask(self, mask, ax, random_color=False):
        """
        Display a mask on a matplotlib axis.
        
        Args:
            mask: The binary mask to display
            ax: Matplotlib axis to display on
            random_color: Whether to use a random color for the mask
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def process_frame(self, frame, frame_idx, save=True):
        """
        Process a single frame with the segmentation model.
        
        Args:
            frame: The image frame to process
            frame_idx: Index of the frame (for saving)
            save: Whether to save the output frame
            
        Returns:
            Processed frame with masks applied
        """
        # Convert frame to PIL Image
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Generate masks
        outputs = self.generator(pil_frame, points_per_batch=self.points_per_batch)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(pil_frame))
        ax = plt.gca()
        
        # Apply all masks
        for mask in outputs["masks"]:
            self.show_mask(mask, ax=ax, random_color=True)
        
        plt.axis("off")
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.output_dir}/frame_{frame_idx:04d}.png", bbox_inches='tight', pad_inches=0)
        
        # Convert the matplotlib figure to an image
        plt.savefig(f"{self.output_dir}/temp.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Load the saved image to return it
        processed_frame = cv2.imread(f"{self.output_dir}/temp.png")
        return processed_frame
        
    def process_video(self, input_path, output_path=None, frame_rate=None):
        """
        Process a video file and apply segmentation to each frame.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video (default: output_dir/segmented_video.mp4)
            frame_rate: Frame rate for output video (default: same as input)
            
        Returns:
            Path to the saved output video
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "segmented_video.mp4")
            
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_rate is None:
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
        try:
            # Process each frame
            frame_idx = 0
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process the frame
                    processed_frame = self.process_frame(frame, frame_idx)
                    
                    # Resize processed frame to match original dimensions if needed
                    if processed_frame.shape[:2] != (height, width):
                        processed_frame = cv2.resize(processed_frame, (width, height))
                    
                    # Write to output video
                    out.write(processed_frame)
                    
                    frame_idx += 1
                    pbar.update(1)
            
            print(f"Video processing complete. Output saved to {output_path}")
            return output_path
            
        finally:
            # Release resources
            cap.release()
            out.release()
            # Remove temporary file
            if os.path.exists(f"{self.output_dir}/temp.png"):
                os.remove(f"{self.output_dir}/temp.png")

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = VideoSegmentationPipeline(device=0, points_per_batch=4)
    
    # Process a video file
    input_video = "gear_video.mp4"
    output_video = "output/segmented_video.mp4"
    
    pipeline.process_video(input_video, output_video)