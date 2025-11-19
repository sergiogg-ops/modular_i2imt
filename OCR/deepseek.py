from transformers import AutoModel, AutoTokenizer, logging as hf_logging
from argparse import ArgumentParser
from json import loads
from tqdm import tqdm
import torch
import os
import re
import yaml
import sys
from contextlib import contextmanager

hf_logging.set_verbosity_error()

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def parse_args():
    parser = ArgumentParser(description="DeepSeek OCR Model")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    parser.add_argument("-m", "--model", default="deepseek-ai/DeepSeek-OCR", help="Pretrained model name")
    parser.add_argument("-gpu", "--use_gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def horizontal_bbox(box):
    x_min, y_min, x_max, y_max = box
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def forward(model, tokenizer, image_path, output_path):
    prompt = "<image>\n<|grounding|>OCR this image."

    with suppress_stdout():
        res = model.infer(tokenizer, 
                    prompt=prompt, 
                    image_file=image_path, 
                    output_path='model_outputs', 
                    base_size=1024, 
                    image_size=512, 
                    crop_mode=True,
                    eval_mode=True)
    texts = re.findall(r'<\|ref\|>(.+?)<\|/ref\|>', res)
    #texts = ' '.join(texts)
    bboxes = re.findall(r'<\|det\|>(.+?)<\|/det\|>', res)
    bboxes = [horizontal_bbox(loads(bbox)[0]) for bbox in bboxes]
    return texts, bboxes
    

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, 
                                      #_attn_implementation='flash_attention_2', 
                                      trust_remote_code=True, 
                                      use_safetensors=True)
    model = model.eval()
    if torch.cuda.is_available() and args.use_gpu:
        model = model.cuda()
    model = model.to(torch.bfloat16)

    results = {}
    for image_path in tqdm(args.paths, unit="image"):
        texts, bboxes = forward(model, tokenizer, image_path, args.output)
        results[os.path.basename(image_path)] = {"text": texts, "bboxes": bboxes}
    
    with open(args.output, 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()