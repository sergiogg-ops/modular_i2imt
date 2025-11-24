from modelscope.pipelines import pipeline
from argparse import ArgumentParser
from tqdm import tqdm
from shapely.geometry import Polygon
import os
import torch
import yaml
import numpy as np
import cv2

READER, MT_MODEL, MT_TOKENIZER, IMG_MODEL = None, None, None, None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MASK_URL = '/tmp/mask_img.jpg'
# Fix for numpy deprecation of np.int0
if not hasattr(np, 'int0'):
    np.int0 = np.int32

def parse_args():
    parser = ArgumentParser(description="Generate images with modified text using AnyText model")
    parser.add_argument("input", type=str, nargs='+', help="Path to the input images.")
    parser.add_argument("-box","--bboxes", type=str, help="Path to the bbox file containing the bounding boxes for the texts.")
    parser.add_argument("-t","--text", type=str, required=True, help="Path to the text file containing the target texts and their bounding boxes.")
    parser.add_argument("--output", type=str, help="Path to the output directory to save results.")
    parser.add_argument("--model", type=str, default="iic/cv_anytext_text_generation_editing", help="Path to the image model for generating images.")
    parser.add_argument("--config", type=str, default="models_yaml/anytext_sd15.yaml", help="Path to the configuration file for the image model.")
    parser.add_argument("--font", type=str, default="font/Arial_Unicode.ttf", help="Path to the font file to use for rendering text.")
    parser.add_argument("--margin", type=int, default=2, help="Margin to add when removing bbox overlaps.")
    return parser.parse_args()

def difference(poly1, poly2, margin=0):
    """
    Compute the geometric difference between two polygons. They must be axis-aligned rectangles.
    Args:
        poly1: First polygon as shapely Polygon object.
        poly2: Second polygon as shapely Polygon object.
    Returns:
        A shapely Polygon representing the area of poly1 minus the area of poly2.
    """
    x1_min, y1_min, x1_max, y1_max = poly1.bounds
    x2_min, y2_min, x2_max, y2_max = poly2.bounds
    
    if y1_max >= y2_min and y1_min < y2_max:
        y1_max = y2_min - margin
    elif y1_min <= y2_max and y1_max > y2_min:
        y1_min = y2_max + margin
    poly1 = Polygon([(x1_min, y1_min), (x1_max, y1_min), (x1_max, y1_max), (x1_min, y1_max)])
    if not poly1.intersects(poly2):
        return poly1
    if x1_max >= x2_min and x1_min < x2_max:
        x1_max = x2_min - margin
    elif x1_min <= x2_max and x1_max > x2_min:
        x1_min = x2_max + margin
    return Polygon([(x1_min, y1_min), (x1_max, y1_min), (x1_max, y1_max), (x1_min, y1_max)])

def remove_bbox_overlaps(bboxes, margin=0):
    """
    Remove overlaps between 2D bounding boxes (polygons).
    Works with arbitrary orientations, not just axis-aligned boxes.
    
    Args:
        bboxes: List of bounding boxes, each represented as: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (4 corners)
        margin: Optional margin to add around subtracted areas
    
    Returns:
        List of modified polygons with overlaps removed.
        Each element is a list of polygons (in case splitting occurs).
    """
    # Convert bboxes to Shapely Polygon objects
    polygons = [Polygon(bbox) for bbox in bboxes]
    #polygons = [poly.buffer(4) for poly in polygons] # Slightly enlarge to avoid precision issues
    
    # Process polygons in order, subtracting overlaps from later polygons
    result_polygons = []
    for i, current_poly in enumerate(polygons):
        remaining = current_poly        
        # Subtract all previous polygons from the current one
        for prev_poly in polygons[:i]:
            if remaining.is_empty:
                break
            #remaining = remaining.difference(prev_poly)
            if remaining.intersects(prev_poly):
                remaining = difference(remaining, prev_poly, margin)
        result_polygons.append(remaining)
    
    # Convert Shapely polygons back to coordinate format
    result = []
    for poly in result_polygons:
        if poly.geom_type == 'Polygon':
            coords = list(poly.exterior.coords[:-1])  # Exclude duplicate last point
            result.append(coords)
    
    return result

def draw_mask(bboxes, image_path, save_path, margin=2):
    # Read image to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    mask = np.ones((height, width, 3), dtype=np.uint8) * 255
    bboxes = remove_bbox_overlaps(bboxes, margin)
    for polygon_coords in bboxes:
        pts = np.array(polygon_coords, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=(0, 0, 0))
    cv2.imwrite(save_path, mask)

    # img = np.array(Image.open(save_path).convert("RGB"))
    # pos = 255 - img          # same inversion
    # pos = pos[..., 0:1]
    # _, pos_bin = cv2.threshold(pos, 254, 255, cv2.THRESH_BINARY)
    # num_labels, labels = cv2.connectedComponents(pos_bin.astype("uint8"))
    # print("positions:", num_labels - 1)  # background excluded


def forward(bboxes, texts, file, margin=2):

    draw_mask(bboxes, file, MASK_URL, margin=margin)
    prompt = 'A poster that reads:'
    for i, t in enumerate(texts):
        prompt += f'\n Line {i+1}: "{t}"'
    input_data = {
        "prompt": prompt,
        "seed": 8943410,
        "draw_pos": MASK_URL,
        "ori_image": file
    }
    results, rtn_code, rtn_warning, _ = IMG_MODEL(input_data, 
                                                    mode='text-editing',
                                                    ddim_steps=20,
                                                    image_count=2,
                                                    show_debug=True)
    if rtn_code != 0:
        print(rtn_warning)
    return results[0][..., ::-1]

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    for img in data.keys():
        bboxes = []
        for bbox in data[img]['bboxes']:
            bboxes.append([(x, y+4) for x, y in bbox])
        data[img]['bboxes'] = bboxes
    return data

def main():
    global READER, MT_MODEL, MT_TOKENIZER, IMG_MODEL
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    data_box = read_file(args.bboxes)
    data_text = read_file(args.text)

    IMG_MODEL = pipeline('my-anytext-task', 
                model=args.model, 
                model_revision='v1.1.1', 
                use_fp16=False, 
                use_translator=False,
                font_path=args.font,
                cfg_path=args.config)
    
    for file in tqdm(args.input, unit="image"):
        name = os.path.basename(file)
        if name not in data_text:
            print(f"Warning: {name} not found in {args.text}. Skipping.")
            continue
        if name not in data_box:
            print(f"Warning: {name} not found in {args.bboxes}. Skipping.")
            continue
        bboxes = data_box[name]['bboxes']
        texts = data_text[name]['text']
        try:
            gen_img = forward(bboxes, texts, file, margin=args.margin)
            if gen_img is not None:
                cv2.imwrite(os.path.join(args.output, os.path.basename(file)), gen_img)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    os.remove(MASK_URL)
    name = args.output if args.output[-1] != '/' else args.output[:-1]

if __name__ == "__main__":
    main()