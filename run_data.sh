#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval.py --cfg config_natural_llava.yaml