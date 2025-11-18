from PIL import Image
from tqdm import tqdm
import os
from sys import argv
import numpy as np

dir = argv[1]

for file in tqdm(os.listdir(dir)):
    filename = os.path.join(dir, file)
    img = np.array(Image.open(filename).convert("RGB"))
    if img.shape[0] !=  512:
        os.remove(filename)