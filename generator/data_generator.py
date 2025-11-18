from PIL import Image, ImageDraw, ImageFont
from argparse import ArgumentParser
from jiwer import wer
from tqdm import tqdm
import numpy as np
import random
import easyocr
import yaml
import os

def parse_args():
    parser = ArgumentParser(description="Create a sinthetic dataset of images with text.")
    parser.add_argument("--images", type=str, help="Path to the input directory containing images.")
    parser.add_argument("--src_text", type=str, help="Path to the source text file containing text to be drawn on images.")
    parser.add_argument("--src_lang", type=str, help="Source language for translation.")
    parser.add_argument("--tgt_text", type=str, help="Path to the target text file containing text to be drawn on images.")
    parser.add_argument("--tgt_lang", type=str, help="Target language for translation.")
    parser.add_argument("--output", type=str, help="Path to the output directory to save results.")
    parser.add_argument("--font_path", type=str, nargs='+', help="Path to the font file to use for drawing text.")
    parser.add_argument("--min_font_height", type=int, default=60, help="Minimum height of the font to use for drawing text.")
    parser.add_argument("--max_font_height", type=int, default=60, help="Maximum height of the font to use for drawing text.")
    parser.add_argument("--min_slope", type=int, default=-0, help="Minimum slope for the text.")
    parser.add_argument("--max_slope", type=int, default=0, help="Maximum slope for the text.")
    parser.add_argument("--min_width", type=int, default=256, help="Minimum width of the text bounding box.")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate.")
    parser.add_argument("--curation_threshold", type=float, default=0.2, help="Threshold for curation of generated images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, help="Path to the configuration file in YAML format.")

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
        args.images = config.get('images', args.images)
        args.src_text = config.get('src_text', args.src_text)
        args.src_lang = config.get('src_lang', args.src_lang)
        args.tgt_text = config.get('tgt_text', args.tgt_text)
        args.tgt_lang = config.get('tgt_lang', args.tgt_lang)
        args.output = config.get('output', args.output)
        args.font_path = config.get('font_path', args.font_path)
        args.min_font_height = config.get('min_font_height', args.min_font_height)
        args.max_font_height = config.get('max_font_height', args.max_font_height)
        args.min_slope = config.get('min_slope', args.min_slope)
        args.max_slope = config.get('max_slope', args.max_slope)
        args.min_width = config.get('min_width', args.min_width)
        args.num_images = config.get('num_images', args.num_images)
        args.curation_threshold = config.get('curation_threshold', args.curation_threshold)
        args.seed = config.get('seed', args.seed)
    return args

def load_config(config_path: str):
    '''
    Load configuration from a YAML file.
    '''
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class Std_Text(object):
    def __init__(self, font_path, font_height):
        self.border_width = 10
        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.font = ImageFont.truetype(font_path, font_height)

    def draw_text(self, text, image, bbox, slope):
        if not text:
            return image, np.array([])
        x1, y1, x2, y2 = bbox
        max_width = x2 - x1

        # Calculate the size needed for the text before rotation
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1))) # dummy draw object
        wrapped_text, widths = self.wrap_text_pillow(draw, text, max_width)
        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=self.font)
        text_width = text_bbox[2]
        text_height = text_bbox[3]

        # Create a new, transparent image to draw the text on.
        # Use RGBA mode to include the alpha channel for transparency.
        aux_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
        draw_text = ImageDraw.Draw(aux_img)

        # Draw the wrapped text on the transparent canvas
        draw_text.multiline_text((0, 0), wrapped_text, fill=(0, 0, 0, 255), font=self.font)

        # Rotate the text image using the specified slope
        rotated_text_img = aux_img.rotate(slope, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(rotated_text_img, (x1, y1), rotated_text_img)

        # bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        line_height = text_height // len(widths)
        bboxes = [np.array([[x1, y1+line_height*(line-1)], 
                            [x1+w, y1+line_height*(line-1)], 
                            [x1+w, y1+line_height*line], 
                            [x1, y1+line_height*line]]) for w, line in zip(widths, np.arange(1,len(widths)+1))]
        angle_rad = np.radians(-slope)  # negative because PIL rotates counter-clockwise
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Translate bbox to origin (x1, y1)
        bbox_translated = [bbox - np.array([x1, y1]) for bbox in bboxes]
        # Apply rotation matrix
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        bbox_rotated = [np.dot(bbox, rotation_matrix.T) for bbox in bbox_translated]
        # Translate back
        bboxes = [bbox + np.array([x1, y1]) for bbox in bbox_rotated]

        return image, bboxes

    def wrap_text_pillow(self, draw, text, max_width):
        # Your existing text wrapping logic
        words = text.split()
        lines = []
        widths = []
        cur = ""
        for w in words:
            test = w if cur == "" else cur + " " + w
            # Use textlength for accurate width calculation
            w_width = draw.textlength(test, font=self.font)
            if w_width <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                    widths.append(draw.textlength(cur, font=self.font))
                cur = w
        if cur:
            lines.append(cur)
            widths.append(draw.textlength(cur, font=self.font))
        lines = '\n'.join(lines)
        return lines, widths

def generate_pair(images, src_texts, tgt_texts, fonts, min_height, max_height, min_slope, max_slope, min_width):
    '''
    Generate a pair (source and target) of images with text.

    Args:
        images (list): List of image file paths.
        src_texts (list): List of source text strings.
        tgt_texts (list): List of target text strings.
        fonts (list): List of font file paths.
        min_height (int): Minimum height of the text bounding box.
        max_height (int): Maximum height of the text bounding box.
        min_slope (int): Minimum slope for the text.
        max_slope (int): Maximum slope for the text.
        min_width (int): Minimum width of the text bounding box.

    Returns:
        PIL.Image: Source image with text.
        PIL.Image: Target image with text.
        str: Source text.
        str: Target text.
    '''
    # Select the content of the image
    bg_image = random.choice(images)
    bg_image = Image.open(bg_image).convert("RGB")
    txt_idx = random.randint(0, len(src_texts) - 1)
    src_text = src_texts[txt_idx]
    tgt_text = tgt_texts[txt_idx]

    # Select visual parameters
    size = bg_image.size
    height = random.randint(min_height, max_height)
    slope = random.randint(min_slope, max_slope)
    bbox = [random.randint(0, size[0] - min_width), random.randint(0, size[1] - height)]
    bbox.append(bbox[0] + random.randint(min_width, size[0] - bbox[0]))
    bbox.append(bbox[1] + random.randint(height, size[1] - bbox[1]))
    font = random.choice(fonts)

    # Draw images
    drawer = Std_Text(font, height)
    src_image, src_bboxes = drawer.draw_text(src_text, bg_image.copy(), bbox, slope)
    tgt_image, tgt_bboxes = drawer.draw_text(tgt_text, bg_image.copy(), bbox, slope)
    return src_image, tgt_image, src_text, tgt_text, src_bboxes, tgt_bboxes

def verify(reader, img, text, threshold):
    '''
    Verify if the text is correctly drawn on the image depending on the metric between the ocr and the reference text.
    
    Args:
        reader (easyocr.Reader): The OCR reader to use for text recognition.
        metric (str): The metric to use for verification (e.g., WER, ...).
        img (PIL.Image): The image to verify.
        text (str): The text that should be present in the image.
        threshold (float): The threshold for the metric to consider the verification successful.

    Returns:
        bool: True if the text is correctly drawn, False otherwise.
    '''
    hyp = reader.readtext(np.array(img), paragraph=True)
    hyp = " ".join([x[1] for x in hyp])
    score = wer(hyp, text)
    return score < threshold

def create_dirs(output, lang):
    '''
    Create directories for the output images and text files.
    
    Args:
        output (str): The base output directory.
        lang (str): The language for which to create the directories.
    '''
    if os.path.exists(f'{output}_{lang}'):
        for file in os.listdir(f'{output}_{lang}'):
            os.remove(os.path.join(f'{output}_{lang}', file))
    else:
        os.makedirs(f'{output}_{lang}')
    if os.path.exists(f'{output}.{lang}'):
        os.remove(f'{output}.{lang}')

def main():
    args = parse_args()
    create_dirs(args.output, args.src_lang)
    create_dirs(args.output, args.tgt_lang)
    random.seed(args.seed)

    # Load images
    images = [os.path.join(args.images, img) for img in os.listdir(args.images) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    if not images:
        raise ValueError("No images found in the specified directory.")
    # Load source and target texts
    with open(args.src_text, 'r') as f:
        src_texts = f.read().splitlines()
    with open(args.tgt_text, 'r') as f:
        tgt_texts = f.read().splitlines()
    if not src_texts or not tgt_texts:
        raise ValueError("Source or target text files are empty.")
    
    src_reader = easyocr.Reader([args.src_lang],gpu=False)
    tgt_reader = easyocr.Reader([args.tgt_lang],gpu=False)
    
    print("Starting dataset generation...")
    print(f"Images will be saved to {args.output}_{args.src_lang} and {args.output}_{args.tgt_lang}")
    count = 0
    src_meta, tgt_meta = [], []
    progress = tqdm(total=args.num_images, desc="Generating", unit="pair")
    while count < args.num_images:
        src_image, tgt_image, src_text, tgt_text, src_bboxes, tgt_bboxes = generate_pair(
            images, src_texts, tgt_texts, args.font_path, 
            args.min_font_height, args.max_font_height, 
            args.min_slope, args.max_slope, args.min_width
        )

        if verify(src_reader, src_image, src_text, args.curation_threshold) and \
           verify(tgt_reader, tgt_image, tgt_text, args.curation_threshold):
            # Save images
            src_image.save(os.path.join(f'{args.output}_{args.src_lang}', f'{count}.png'))
            tgt_image.save(os.path.join(f'{args.output}_{args.tgt_lang}', f'{count}.png'))

            # Save metadata
            src_meta.append({
                'image': f'{count}.png',
                'text': src_text,
                'bboxes': [bbox.tolist() for bbox in src_bboxes]
            })
            tgt_meta.append({
                'image': f'{count}.png',
                'text': tgt_text,
                'bboxes': [bbox.tolist() for bbox in tgt_bboxes]
            })

            count += 1
            progress.update(1)
    progress.close()
    # Save metadata to files
    print("Saving metadata, please don't interrupt the process...")
    with open(f'{args.output}.{args.src_lang}', 'w', encoding='utf-8') as src_file:
        yaml.dump(src_meta, src_file, allow_unicode=True)
    with open(f'{args.output}.{args.tgt_lang}', 'w', encoding='utf-8') as tgt_file:
        yaml.dump(tgt_meta, tgt_file, allow_unicode=True)
    print(f"Metadata stored in {args.output}.{args.src_lang} and {args.output}.{args.tgt_lang}")
    print("Dataset generation completed.")

if __name__ == "__main__":
    main()