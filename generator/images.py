import numpy as np
from PIL import Image
import os

NUM_IMAGES = 500
DIR = "images"
def generate_image(index):
    color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
    random_array = np.ones((512, 512, 3), dtype=np.uint8) * color
    img = Image.fromarray(random_array, 'RGB')
    img.save(f"{DIR}/image_{index}.png")

if __name__ == "__main__":
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    for i in range(NUM_IMAGES):
        generate_image(i)
