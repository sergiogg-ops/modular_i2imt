from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from argparse import ArgumentParser
from os.path import basename
from tqdm import tqdm
import os
import yaml
import torch

def parse_args():
    parser = ArgumentParser(description="OCR using Doctr's TrOCR model")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    parser.add_argument("-gpu", "--use_gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def read_image(image_path):
    return DocumentFile.from_images(image_path)

def horizontal_bbox(box):
    x_min, y_min, x_max, y_max = box
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def parse_doctr(document, img_size=512):
    texts = []
    bboxes = []
    for line in document.pages[0].blocks[0].lines:
        bbox = line.geometry
        bbox = [[float(box[0]*img_size), float(box[1]*img_size)] for box in bbox]
        bbox = horizontal_bbox([bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]])
        text = ''
        for word in line.words:
            text += word.value + ' '
        texts.append(text.strip())
        bboxes.append(bbox)
    return texts, bboxes

def forward(predictor, document):
    document = predictor(document)
    return parse_doctr(document)

def main():
    args = parse_args()
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    predictor = ocr_predictor(pretrained=True).to(device)

    results = {}
    for image_path in tqdm(args.paths, unit="image"):
        document = read_image(image_path)
        texts, bboxes = forward(predictor, document)
        #combined_text = ' '.join([' '.join(block) for block in texts])
        results[basename(image_path)] = {
            "text": texts,
            "bboxes": bboxes
        }
    with open(args.output, "w") as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()