import easyocr
import yaml
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from os.path import basename

def parse_args():
    parser = ArgumentParser(description="OCR using EasyOCR")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-l", "--lang", default="en", help="Language for OCR")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    return parser.parse_args()

def read_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))

def forward(reader, image):
    detection = reader.readtext(image, paragraph=False)
    text = ' '.join([det[1] for det in detection])
    bbox = [[[int(n) for n in point] for point in det[0]] for det in detection]
    return bbox, text

def main():
    args = parse_args()
    reader = easyocr.Reader([args.lang], gpu=False)

    results = []
    for image_path in args.paths:
        image = read_image(image_path)
        text = forward(reader, image)
        results.append({
            "image_path": basename(image_path),
            "text": text[1],
            "bbox": text[0]
        })
    with open(args.output, "w") as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()