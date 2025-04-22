#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python data_test.py --cfg config_natural_llava.yaml