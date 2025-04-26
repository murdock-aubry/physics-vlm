# 📘 Measuring Vision Language Models' Ability to Detect Physical Anomalies

This repository contains the official implementation for the paper:  
**_Measuring Vision Language Models’ Ability to Detect Physical Anomalies_**

This work explores the extent to which modern **Vision-Language Models (VLMs)** can perceive and reason about physical phenomena in dynamic environments. Our experiments probe whether these models adhere to intuitive physical laws such as **object permanence** and **spatio-temporal continuity**, using controlled video datasets and structured prompting strategies.

> This work builds directly off the work done in the: [vjepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics) repository from Facebook Research. While the datasets and problem framing are similar, this repository focuses specifically on probing **VLMs** and their text-based understanding of physical violations.

---

## 🛠 Getting Started

We recommend using [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) to manage dependencies for reproducibility.

### Step 1: Create and activate environment

```bash
micromamba create -n physical-vlm python=3.10 pip
micromamba activate physical-vlm
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Datasets

This codebase supports two datasets for evaluating physical reasoning:

1. **[IntPhys Dataset](https://intphys.cognitive-ml.fr/)**  
2. **[InfLevel Dataset](https://github.com/allenai/inflevel)**

Please make sure to download and structure the dataset directories according to your setup. The paths should match what’s specified in the config files or script arguments.

---

## 🔽 Downloading Pretrained Models

You can the pretrained models used can be viewed on Hugging Face:

- **[Video-LLaVA-7B-natural-2000](https://huggingface.co/mvrdock/Video-LLaVA-7B-natural-2000)**
- **[Video-LLaVA-7B-natural](https://huggingface.co/mvrdock/Video-LLaVA-7B-natural)**

These models are necessary for running inference and fine-tuning tasks.

---

## 🎯 Fine-Tuning the Vision-Language Model

To fine-tune the VLM using **LoRA**, run:

```bash
python finetune_lora.py
```

Before running, **edit the following variables** inside `finetune_lora.py`:

- `output_dir`: directory to save fine-tuned model checkpoints
- `data_path`: path to your dataset (IntPhys or InfLevel)

---

## 🔍 Running Inference

### Option 1: Inference on a Single Video

```bash
python test_video_inputs.py
```

This script allows running inference on one video clip using the following response modes:

- `binary_resp`: binary yes/no response to physical violations
- `continuous_resp`: scalar confidence score
- `cot_resp`: chain-of-thought reasoning

At the bottom of the script, you'll find dataset-specific loading instructions. Adjust paths and function calls as needed.

---

### Option 2: Full Dataset Evaluation

```bash
python eval.py --cfg path/to/your_config.yaml
```

Or via the shell wrapper:

```bash
bash run_data.sh
```

#### Configs:
- `config_natural_llaval.yaml`: for the **IntPhys** dataset  
- `config_natural_llava_inflevel.yaml`: for the **InfLevel** dataset

These YAML files define dataset paths, model configs, prompting templates, and output formats.

---

## 🧠 Prompting

A **`prompt.yaml`** file is included in the repository, which contains all the prompts used in the paper in their full form. This file can be used to understand the different prompting strategies employed for detecting physical anomalies and evaluating model responses.

---

## 📊 Output and Analysis

Outputs are saved in the configured `output_dir`. These include:

- Raw inference results
- Aggregated scores by physical concept
- **Jupyter Notebooks** to reproduce all graphs in the paper

---

## 🧪 Miscellaneous

### `sample_videos_inflevel/`

Contains:

- Tables of start times for physical concept tests (used for video slicing)
- The script `extract_start_time.py` for reproducing evaluation windows from the **vjepa** dataset

---

## 📄 Paper

The official paper can be found here: [View the final report](./final_report.pdf)
```
