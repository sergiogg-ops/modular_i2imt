from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from argparse import ArgumentParser
import yaml

def parse_args():
    parser = ArgumentParser(description="OCR using Doctr's TrOCR model")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    return parser.parse_args()

def read_image(image_path):
    return DocumentFile.from_images(image_path)

def parse_doctr(document, img_size=512):
    texts = []
    bboxes = []
    for block in document.pages[0].blocks:
        texts.append([])
        bboxes.append([])
        for line in block.lines:
            bbox = line.geometry
            bbox = [[box[0]*img_size, box[1]*img_size] for box in bbox]
            text = ''
            for word in line.words:
                text += word.value + ' '
            texts[-1].append(text.strip())
            bboxes[-1].append(bbox)
    return texts, bboxes

def forward(predictor, document):
    document = predictor(document)
    return parse_doctr(document)

def main():
    args = parse_args()
    predictor = ocr_predictor(pretrained=True)

    results = []
    for image_path in args.paths:
        document = read_image(image_path)
        texts, bboxes = forward(predictor, document)
        combined_text = ' '.join([' '.join(block) for block in texts])
        combined_bboxes = [bbox for block in bboxes for bbox in block]
        results.append({
            "image_path": image_path.split('/')[-1],
            "text": combined_text,
            "bbox": combined_bboxes
        })
    with open(args.output, "w") as f:
        yaml.dump(results, f)