from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from argparse import Namespace, ArgumentParser
import torch
import numpy as np
import random 
from tqdm import tqdm
import yaml
import os
import pandas as pd

#os.environ["HF_HOME"] = "/w/340/abarroso/huggingface"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Flatten experiment-specific config into top level
    experiment = cfg_dict["experiment_name"]
    experiment_fields = cfg_dict["experiment_config"][experiment]["fields"]
    
    cfg_dict["experiment_fields"] = experiment_fields
    return Namespace(**cfg_dict)

def get_dataset(dataset_type, video_dir, processor, prompt, experiment_fields, n, start_times):
    if dataset_type == "inflevel":
        from inlfevel_dataset import VideoDataset
        return VideoDataset(video_dir, processor, prompt, experiment_fields=experiment_fields, start_times=start_times,  n=n)  # Adjust parameters as needed
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def parse_response_value(output):
    """
    Parses the model's output into a numeric value.
    If the output is a valid number (int/float), it returns that number.
    If not, it returns 5.0 as the default value.
    """
    try:
        # Try to convert the output to a float
        return float(output)
    except ValueError:
        # If it can't be converted, return the default value (5.0)
        return 5.0

def main(config):
    
    set_seeds()
    os.environ["HF_HOME"] = config.hf_home
    cache_dir = os.path.join(config.hf_home, "models")
    video_dir = os.path.join(config.video_dir, config.experiment_name)
    loaded_df = pd.read_csv(f"{config.experiment_name}_start_frame.csv",index_col='video_path')
    start_times = loaded_df.to_dict(orient='index')
    # Define the dataset and data loader
    base_model = VideoLlavaForConditionalGeneration.from_pretrained(
        config.base_model_name,
        load_in_4bit=True,
        cache_dir=cache_dir
    )
    # Only load adapter if specified
    if config.adapter_name is not None:
        model = PeftModel.from_pretrained(base_model, config.adapter_name, cache_dir=cache_dir)
    else:
        model = base_model  # Use base model directly
    processor = VideoLlavaProcessor.from_pretrained(config.base_model_name, cache_dir=cache_dir)
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "uniform"


    dataset = get_dataset(config.dataset_type, video_dir, processor,prompt=config.prompt, experiment_fields=config.experiment_fields, n=config.n_frames, start_times=start_times)
    from inlfevel_dataset import collate_fn_factory
    collate_fn = collate_fn_factory(config.experiment_fields)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)  # Adjust num_workers for parallelization

    # Process all videos in the dataset
    results = []

    # Early stopping condition after processing 20 videos
    #max_files = 10
    #files_processed = 0

    #reinsert dir for continuity
    for inputs, video_file, metadata_dict in tqdm(dataloader, desc="Processing videos"):
        # Move inputs to the model device (GPU)
        #if files_processed >= max_files:
        #    print(f"Early stopping after processing {max_files} files.")
        #    break
        #inputs = {k: v.squeeze(0).to(model.device) for k, v in inputs.items()}
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Generate the model response
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False  # Use greedy decoding
        )
        
        # Slice off input part and decode the response
        response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        # Append results with metadata
        batch_results = [] 
        for i in range(len(video_file)):
            response_value = parse_response_value(output[i])
            
            video_result = {
                "video_name": video_file[i],  # Assuming video_file is iterable
                "response": output[i],  # Assuming output is iterable for batch size
                "response_value": response_value
            }

            # Add metadata for each batch item
            for field, value in metadata_dict.items():
                # Handling metadata fields for batch size > 1
                video_result[field] = value[i] if len(value) > 1 else value[0]  # Select the right metadata for each item in batch

            batch_results.append(video_result)
        
        results.extend(batch_results)
        #files_processed += 1  # Increment the counter
    # Convert results to a pandas DataFrame and save to a CSV file
    print(results)
    df = pd.DataFrame(results)
    print(df)
    output_csv = f"video_responses_{config.output_file_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Processing complete. Results saved to '{output_csv}'.")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.cfg)
    main(config)