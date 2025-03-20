from datasets import load_dataset
import numpy as np


ds = load_dataset("wofmanaf/ego4d-video")

idx = 40
video = ds["train"]['video'][idx]

print(np.load(ds["train"]["video"][1]))