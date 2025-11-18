from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
from json import loads
import torch
import os
import re
import yaml

def parse_args():
    parser = ArgumentParser(description="DeepSeek OCR Model")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    parser.add_argument("-m", "--model", default="deepseek-ai/DeepSeek-OCR", help="Pretrained model name")
    return parser.parse_args()

def horizontal_bbox(box):
    x_min, y_min, x_max, y_max = box
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

def forward(model, tokenizer, image_path, output_path):
    prompt = "<image>\n<|grounding|>OCR this image."

    res = model.infer(tokenizer, 
                  prompt=prompt, 
                  image_file=image_path, 
                  output_path='model_outputs', 
                  base_size=1024, 
                  image_size=512, 
                  crop_mode=True,
                  eval_mode=True)
    texts = re.findall(r'<\|ref\|>(.+?)<\|/ref\|>', res)
    texts = ' '.join(texts)
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
    model = model.eval().cuda().to(torch.bfloat16)

    results = {}
    for image_path in args.paths:
        texts, bboxes = forward(model, tokenizer, image_path, args.output)
        results[os.basename(image_path)] = {"text": texts, "bboxes": bboxes}
    
    with open(args.output, 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main()