from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace, ArgumentParser
import torch
import numpy as np
import random 
from tqdm import tqdm
import yaml
import os
import pandas as pd
from data_inf_fns import inflevel_inference, intphys_binary_resp, intphys_continuous_resp, intphys_cot_resp
from functools import partial

#os.environ["HF_HOME"] = "/w/340/abarroso/huggingface"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Flatten experiment-specific config into top level
    if cfg_dict["dataset_type"] == "inflevel":
        experiment = cfg_dict["experiment_name"]
        experiment_fields = cfg_dict["experiment_config"][experiment]["fields"]
        cfg_dict["experiment_fields"] = experiment_fields
    return Namespace(**cfg_dict)

def get_dataset(dataset_type, video_dir, processor, prompt, experiment_fields, n, experiment_name):
    if dataset_type == "inflevel":
        from phys_datasets import Inflevel_Dataset, Inflevel_collate_fn_factory
        collate_fn = Inflevel_collate_fn_factory(experiment_fields)
        loaded_df = pd.read_csv(f"inflevel_data/{experiment_name}_start_frame.csv",index_col='video_path')
        start_times = loaded_df.to_dict(orient='index')
        return Inflevel_Dataset(video_dir, processor, prompt, experiment_fields=experiment_fields, start_times=start_times,  n=n), collate_fn  # Adjust parameters as needed
    elif dataset_type == "intphys":
        from phys_datasets import IntPhysDataset
        return IntPhysDataset(video_dir, processor, prompt, n), None
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
def get_processing_function(config):
    if config.dataset_type == "inflevel":
        return inflevel_inference
    elif config.dataset_type == "intphys":
        if config.test_type=="binary":
            return intphys_binary_resp
        elif config.test_type=="continuous":
            return intphys_continuous_resp
        elif config.test_type=="COT":
            llm_model = AutoModelForCausalLM.from_pretrained(config.cot_config['LLM_model_name'], cache_dir = config.cache_dir)
            llm_tokenizer = AutoTokenizer.from_pretrained(config.cot_config['LLM_model_name'], cache_dir = config.cache_dir)
            return partial(intphys_cot_resp, llm_model = llm_model, llm_tokenizer= llm_tokenizer, llm_prompt = config.cot_config['LLM_prompt'])
        else: 
            raise ValueError(f"Unknown test type: {config.test_type}")
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

    


def main(config):
    
    set_seeds()
    os.environ["HF_HOME"] = config.hf_home
    config.cache_dir = os.path.join(config.hf_home, "models")
    video_dir = os.path.join(config.video_dir, config.experiment_name)

    # Define the dataset and data loader
    base_model = VideoLlavaForConditionalGeneration.from_pretrained(
        config.base_model_name,
        load_in_4bit=True,
        cache_dir= config.cache_dir 
    )
    
    # Only load adapter if specified
    if config.adapter_name is not None:
        model = PeftModel.from_pretrained(base_model, config.adapter_name, cache_dir= config.cache_dir )
    else:
        model = base_model  # Use base model directly
        
    processor = VideoLlavaProcessor.from_pretrained(config.base_model_name, cache_dir= config.cache_dir )
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "uniform"
    experiment_fields = getattr(config, "experiment_fields", None)

    dataset, collate_fn = get_dataset(config.dataset_type, video_dir, processor,prompt=config.prompt, experiment_fields=experiment_fields, n=config.n_frames, experiment_name=config.experiment_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)  # Adjust num_workers for parallelization

    results = []
    results_function = get_processing_function(config)
    #count = 0
    for inputs, video_file, metadata in tqdm(dataloader, desc="Processing videos"):
        batch_results = results_function(inputs, video_file, metadata, model, processor)
        results.extend(batch_results)
        #count +=1
        #if count == 3: break 
        
    df = pd.DataFrame(results)
    print(df)
    if config.dataset_type == "intphys":
        output_csv = f"outputs/resp_{config.dataset_type}_{config.experiment_name}_{config.test_type}_{config.output_file_name}.csv"
    else: 
        output_csv = f"outputs/resp_{config.dataset_type}_{config.experiment_name}_{config.output_file_name}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Processing complete. Results saved to '{output_csv}'.")
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.cfg)
    main(config)