import os
import json
import random
import numpy as np
import torch
import av
from torch.utils.data import Dataset, DataLoader
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Video reading function
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
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

# Custom Dataset for video-caption pairs
class VideoCaptionDataset(Dataset):
    def __init__(self, data_path, processor, num_frames=8, num_samples = 100):
        """
        Args:
            data_path (str): Path to JSON file with video paths and captions
            processor: VideoLlavaProcessor instance
            num_frames (int): Number of frames to sample from each video
        """
        self.processor = processor
        self.num_frames = num_frames
        
        # Load the dataset JSON file
        # with open(data_path, 'r') as f:
        #     self.data = json.load(f)

        full_dataset = load_dataset(data_path, split="train")
        indices = np.random.choice(len(full_dataset), size=num_samples, replace=False)
        self.data = full_dataset.select(indices)
        del full_dataset
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item['contentUrl']
        caption = item['name']

        print(video_path, flush = True)

        
        
        # Format prompt
        prompt = f"USER: <video>\nDescribe the video. ASSISTANT: {caption}"
        
        try:
            # Read video frames
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            video_clip = read_video_pyav(container, indices)
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                videos=video_clip,
                return_tensors="pt",
                padding="max_length",
                truncation=True
            )
            
            # Remove batch dimension
            for k, v in inputs.items():
                inputs[k] = v.squeeze(0)
            
            # Prepare labels for language modeling
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Mask input portion for loss calculation (we only want loss on the response)
            assistant_prefix = "ASSISTANT:"
            input_text = self.processor.tokenizer.decode(inputs["input_ids"])
            assistant_start = input_text.find(assistant_prefix) + len(assistant_prefix)
            tokenized_assistant_start = len(self.processor.tokenizer(input_text[:assistant_start], 
                                                              add_special_tokens=False).input_ids)
            
            # Set labels to -100 for input portion to ignore in loss calculation
            inputs["labels"][:tokenized_assistant_start] = -100
            
            return inputs
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return a dummy sample in case of error
            return self.__getitem__(0) if idx != 0 else self.__getitem__(1)

# Custom data collator to handle the variable length
def custom_collator(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Get other pixel values or video features
    pixel_values_videos = torch.stack([item["pixel_values_videos"] for item in batch])
    
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        "pixel_values_videos": pixel_values_videos
    }

def main():
    # Configuration
    model_id = "LanguageBind/Video-LLaVA-7B-hf"
        #ADD YOUR DATAPATH HERE
    output_dir = "/projects/dynamics/vlm-tmp"
    data_path = "Mouwiya/Video-10M"

    # target_weights = ["q_proj", "v_proj", "k_proj", "out_proj"]
    # target_weights = ["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    # target_weights = ["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "layer_norm1", "layer_norm2"]

    vision_target_modules = []
    # Add vision encoder layers
    num_layers = 23  # Adjust this based on the actual number of layers
    for i in range(num_layers):
        # Add attention components
        vision_target_modules.extend([
            f"video_tower.vision_model.encoder.layers.{i}.self_attn.q_proj",
            f"video_tower.vision_model.encoder.layers.{i}.self_attn.k_proj",
            f"video_tower.vision_model.encoder.layers.{i}.self_attn.v_proj",
            f"video_tower.vision_model.encoder.layers.{i}.self_attn.out_proj",
            # Add MLP components
            f"video_tower.vision_model.encoder.layers.{i}.mlp.fc1",
            f"video_tower.vision_model.encoder.layers.{i}.mlp.fc2",
        ])

    # Add language model components
    language_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    # language_target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # Combine both sets of target modules
    target_weights = vision_target_modules + language_target_modules

    print("Training the following weights:", language_target_modules)



    num_samples = 2000
    epochs = 3
    
    print("Initializing training params", flush = True)

    # Training parameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        # report_to="tensorboard",
        remove_unused_columns=False,
    )

    print("Loading model", flush = True)
    
    # Load model and processor
    processor = VideoLlavaProcessor.from_pretrained(model_id)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "uniform"
    
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto"  # This will automatically offload parts to CPU if needed
    )

    # model.config.use_memory_efficient_attention = True

    print("Model Loaded", flush = True)
    print("Initializing LoRA config", flush = True)

    # Setup LoRA fine-tuning
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_weights,#["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("Freezing weights", flush = True)
    # # Prepare model for training
    # # Freeze vision encoder
    # for param in model.video_tower.vision_model.parameters():
    #     param.requires_grad = False
    

    print("Applying LoRA", flush = True)
    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    

    print("Initializing training set", flush = True)
    # Create dataset and dataloader
    train_dataset = VideoCaptionDataset(data_path, processor, num_samples=num_samples)
    

    print("Initializing trainer", flush = True)
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collator,
    )
    

    print("Starting training", flush = True)
    # Start training
    trainer.train()
    
    ckpt_name = f"{output_dir}/epoch{epochs}_samples{num_samples}"
    # Save the model
    trainer.save_model(ckpt_name)
    processor.save_pretrained(ckpt_name)
    
    print(f"Model and processor saved to {ckpt_name}", flush = True)
    
    # Test the model after fine-tuning

    # test_model(model, processor, test_video_path)

def test_model(model, processor, video_path):
    # Load test video
    container = av.open(video_path)
    
    # Sample frames
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, 8).astype(int)
    clip = read_video_pyav(container, indices)
    
    # Prepare input
    test_prompt = "USER: <video>\nWhat's happening in this video? ASSISTANT:"
    inputs = processor(text=test_prompt, videos=clip, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Print result
    print("Model output:")
    print(processor.batch_decode(generate_ids, skip_special_tokens=True)[0])

if __name__ == "__main__":
    main()