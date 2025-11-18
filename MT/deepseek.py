from transformers import pipeline
from argparse import ArgumentParser
import yaml

def parse_args():
    parser = ArgumentParser(description='Translate the source file using a pretrained model.')
    parser.add_argument('input_file', type=str, required=True, help='Path to the source file')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1", help='Path to the pretrained model')
    parser.add_argument('--output', type=str, default='output.yaml', help='Path to the output file')
    parser.add_argument('-src','--src_lang', type=str, required=True, help='Source language code')
    parser.add_argument('-tgt','--tgt_lang', type=str, required=True, help='Target language code')
    args = parser.parse_args()
    return args

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def main():
    args = parse_args()
    data = read_file(args.input_file)
    translation_pipeline = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1", trust_remote_code=True)

    identifiers = list(data.keys())
    source_sentences = [data[id]['text'] for id in identifiers]
    source_sentences = [f"Translate the following {args.src_lang} sentence into {args.tgt_lang} without further comments:\n{sent}" for sent in source_sentences]
    messages = [{"role": "user", "content": sent} for sent in source_sentences]
    translations = translation_pipeline(messages)
    print(translations[0])
    exit()

    for id, translation in zip(identifiers, translations):
        data[id]['translation'] = translation

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True)

if __name__ == '__main__':
    main()