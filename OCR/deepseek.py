from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
import yaml

def parse_args():
    parser = ArgumentParser(description="DeepSeek OCR Model")
    parser.add_argument("paths", nargs="+", help="Paths to input images")
    parser.add_argument("-o", "--output", default="output.yaml", help="Output YAML file")
    parser.add_argument("-m", "--model", default="deepseek-ai/DeepSeek-OCR", help="Pretrained model name")
    return parser.parse_args()

def forward(model, tokenizer, image_path, output_path):
    prompt = "<image>\n<|grounding|>OCR this image."

    res = model.infer(tokenizer, 
                      prompt=prompt, 
                      image_file=image_path, 
                      output_path = output_path, 
                      base_size = 1024, 
                      image_size = 512, 
                      crop_mode=True, 
                      save_results = True, 
                      test_compress = True)
    print(res)

def main():
    args = parse_args()
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results = []
    for image_path in args.paths:
        forward(model, tokenizer, image_path, args.output)

if __name__ == "__main__":
    main()