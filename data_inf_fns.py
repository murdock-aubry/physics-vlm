import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import re

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
    
def filter_alphabetic(text):
    # This regex will keep only alphabetic characters (both lowercase and uppercase) and spaces
    return re.sub(r'[^a-zA-Z ]+', '', text)
    

def inflevel_inference(inputs, video_file, metadata_dict, model, processor):
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
    return batch_results

def round_softmax(probs, decimal_places=4):
    # Step 1: Round each probability to the desired decimal places
    scale_factor = 10**decimal_places
    rounded = torch.round(probs * scale_factor) / scale_factor
    
    # Step 2: Normalize to ensure they sum to 1
    normalized = rounded / rounded.sum()
    
    # Convert to Python floats to avoid any precision issues
    return normalized.tolist()  # .tolist() to convert tensor to plain list of floats

def intphys_continuous_resp(inputs, video_file, metadata, model, processor):
      # Get the generated token IDs and the first new token
    inputs = {k: v.squeeze(0).to(device) for k, v in inputs.items()}
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
    yes_prob = normalized_probs[0]
    no_prob = normalized_probs[1]
    return [{
        "file": video_file[0],
        "yes_prob": yes_prob,
        "no_prob": no_prob,
        "label": bool(metadata[0])
    }]

def intphys_binary_resp(inputs, video_file, metadata, model, processor):
    inputs = {k: v.squeeze(0).to(device) for k, v in inputs.items()}
    generate_ids = model.generate(
    **inputs, 
    max_new_tokens=10,
    do_sample=False,  # Use greedy decoding
    #output_scores=True,
    #return_dict_in_generate=True
    )
    print("huh")
    response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    #response_ids = generate_ids["sequences"][:, inputs["input_ids"].shape[1]:]
    output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return [{
        "file": video_file[0],
        "response": output,
        "label": bool(metadata[0])
    }]

    
    


def intphys_cot_resp(inputs, video_file, metadata, model, processor, llm_model, llm_tokenizer,llm_prompt):
    inputs = {k: v.squeeze(0).to(device) for k, v in inputs.items()}
    generate_ids = model.generate(
    **inputs, 
    max_new_tokens=200,
    do_sample=False,  # Use greedy decoding
    #output_scores=True,
    #return_dict_in_generate=True
    )
    response_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    #response_ids = generate_ids["sequences"][:, inputs["input_ids"].shape[1]:]
    llava_output = processor.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    llm_prompt = llm_prompt.format(llava_output=llava_output)
    inputs_pythia = llm_tokenizer(llm_prompt, return_tensors="pt").to(llm_model.device)
    pythia_generate_ids = llm_model.generate(
        **inputs_pythia, 
        max_new_tokens=2,  # Limit to a short answer like "yes" or "no"
        do_sample=False,  # Use greedy decoding
    )

    llm_resp_ids = pythia_generate_ids[:, inputs_pythia["input_ids"].shape[1]:]
    llm_response = llm_tokenizer.decode(llm_resp_ids[0], skip_special_tokens=True)
    return [{
        "file": video_file[0],
        "response": filter_alphabetic(llm_response),
        "label": bool(metadata[0])
    }]
